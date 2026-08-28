"""Age bracket classification head for voice attribute inference with temperature calibration and ordinal support."""

import json
import os
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.config import settings
from app.core.logging import logger
from app.schemas.response import AgeBracketEnum, AgeResult


class AgeClassifierHead(nn.Module):
    """Standard MLP classification head for predicting age bracket from voice embeddings."""

    def __init__(self, embedding_dim: int = 192, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.classes: List[AgeBracketEnum] = [
            AgeBracketEnum.AGE_18_30,
            AgeBracketEnum.AGE_31_45,
            AgeBracketEnum.AGE_46_60,
            AgeBracketEnum.AGE_60_PLUS,
        ]

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, len(self.classes)),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw 4-class class logits."""
        return self.classifier(embedding)


class BaselineLinearAgeHead(nn.Module):
    """Baseline linear classification head (192 -> 4)."""

    def __init__(self, embedding_dim: int = 192) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.classes: List[AgeBracketEnum] = [
            AgeBracketEnum.AGE_18_30,
            AgeBracketEnum.AGE_31_45,
            AgeBracketEnum.AGE_46_60,
            AgeBracketEnum.AGE_60_PLUS,
        ]
        self.classifier = nn.Linear(embedding_dim, len(self.classes))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(embedding)


class DeepMLPAgeHead(nn.Module):
    """Deeper MLP classification head (192 -> 128 -> 64 -> 4)."""

    def __init__(self, embedding_dim: int = 192) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.classes: List[AgeBracketEnum] = [
            AgeBracketEnum.AGE_18_30,
            AgeBracketEnum.AGE_31_45,
            AgeBracketEnum.AGE_46_60,
            AgeBracketEnum.AGE_60_PLUS,
        ]
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, len(self.classes)),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(embedding)


class OrdinalAgeHead(nn.Module):
    """Ordinal classification head predicting 3 cumulative age thresholds (P(Age > 30), P(Age > 45), P(Age > 60))."""

    def __init__(self, embedding_dim: int = 192, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.classes: List[AgeBracketEnum] = [
            AgeBracketEnum.AGE_18_30,
            AgeBracketEnum.AGE_31_45,
            AgeBracketEnum.AGE_46_60,
            AgeBracketEnum.AGE_60_PLUS,
        ]
        # 3 binary threshold logits
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass returning 3 threshold logits."""
        return self.classifier(embedding)

    def predict_probabilities(self, embedding: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
        """Convert threshold logits into normalized 4-class bracket probabilities."""
        with torch.no_grad():
            logits = self.classifier(embedding) / max(temperature, 0.1)
            sigmoids = torch.sigmoid(logits).detach().cpu().numpy()  # shape: (batch, 3) or (3,)

        if sigmoids.ndim == 1:
            sigmoids = sigmoids[np.newaxis, :]

        # Cumulative probability decomposition
        p_gt_30 = sigmoids[:, 0]
        p_gt_45 = sigmoids[:, 1]
        p_gt_60 = sigmoids[:, 2]

        p0 = 1.0 - p_gt_30
        p1 = np.maximum(0.0, p_gt_30 - p_gt_45)
        p2 = np.maximum(0.0, p_gt_45 - p_gt_60)
        p3 = p_gt_60

        probs = np.stack([p0, p1, p2, p3], axis=-1)
        # Normalize each row to sum to 1.0
        row_sums = probs.sum(axis=-1, keepdims=True)
        probs = probs / np.maximum(row_sums, 1e-12)
        return probs


class AgeClassifier:
    """Manages age bracket classification model weights, temperature calibration, inference, and thresholding."""

    def __init__(
        self,
        embedding_dim: int = 192,
        weights_path: Optional[Path] = None,
        calibration_path: Optional[Path] = None,
        confidence_threshold: float = settings.AGE_CONFIDENCE_THRESHOLD,
        device: str = settings.DEVICE,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.model: nn.Module = AgeClassifierHead(embedding_dim=embedding_dim).to(self.device)
        self.model.eval()
        self.classes = [
            AgeBracketEnum.AGE_18_30,
            AgeBracketEnum.AGE_31_45,
            AgeBracketEnum.AGE_46_60,
            AgeBracketEnum.AGE_60_PLUS,
        ]

        self.model_available = False
        self.model_source = "unavailable"
        self.model_type = "mlp"  # 'mlp', 'baseline', 'deep_mlp', or 'ordinal'
        self.temperature = 1.0
        self.weights_path = weights_path or (settings.MODEL_WEIGHTS_DIR / "age_head.pt")
        self.calibration_path = calibration_path or (settings.MODEL_WEIGHTS_DIR / "age_calibration.json")

        self._load_weights(self.weights_path)
        self._load_calibration(self.calibration_path)

    def _load_weights(self, weights_path: Path) -> None:
        """Load genuine trained model weights dynamically based on checkpoint structure."""
        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)

                # Detect architecture from state dict keys and tensor shapes
                last_weight_key = [k for k in state_dict.keys() if "weight" in k][-1]
                num_outputs = state_dict[last_weight_key].shape[0]

                if num_outputs == 3:
                    # Ordinal head
                    self.model = OrdinalAgeHead(embedding_dim=self.embedding_dim).to(self.device)
                    self.model_type = "ordinal"
                elif len(state_dict) <= 2:
                    # Baseline linear
                    self.model = BaselineLinearAgeHead(embedding_dim=self.embedding_dim).to(self.device)
                    self.model_type = "baseline"
                elif any("2.weight" in k for k in state_dict.keys()):
                    # Deep MLP
                    self.model = DeepMLPAgeHead(embedding_dim=self.embedding_dim).to(self.device)
                    self.model_type = "deep_mlp"
                else:
                    self.model = AgeClassifierHead(embedding_dim=self.embedding_dim).to(self.device)
                    self.model_type = "mlp"

                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.model_available = True
                self.model_source = "trained"
                logger.info(
                    f"Loaded genuine trained age classifier weights ({self.model_type}) from {weights_path}",
                    extra={"event": "model_loaded", "model": "age_classifier", "model_type": self.model_type, "path": str(weights_path)},
                )
                return
            except Exception as e:
                logger.warning(
                    f"Failed to load trained age weights from {weights_path}: {e}",
                    extra={"event": "model_load_error", "model": "age_classifier", "error": str(e)},
                )

        # No trained weights available
        self.model_available = False
        self.model_source = "unavailable"

        if settings.REQUIRE_TRAINED_MODELS:
            raise FileNotFoundError(
                f"Strict mode enabled (REQUIRE_TRAINED_MODELS=True) but trained weights missing at {weights_path}"
            )

        logger.info(
            "Trained age classifier weights not found. Age predictions will return 'unknown'.",
            extra={"event": "model_unavailable", "model": "age_classifier", "reason": "trained_weights_missing"},
        )

    def _load_calibration(self, calibration_path: Path) -> None:
        """Load learned temperature parameter for confidence calibration."""
        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.temperature = float(data.get("temperature", 1.0))
                logger.info(
                    f"Loaded age confidence calibration: Temperature T={self.temperature:.4f}",
                    extra={"event": "calibration_loaded", "model": "age_classifier", "temperature": self.temperature},
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load age calibration from {calibration_path}: {e}",
                    extra={"event": "calibration_load_error", "model": "age_classifier", "error": str(e)},
                )

    def predict(self, embedding: torch.Tensor) -> AgeResult:
        """Predict age bracket from voice embedding with temperature calibration."""
        if not self.model_available:
            logger.debug(
                "Age inference skipped (model unavailable)",
                extra={
                    "event": "model_unavailable",
                    "model": "age_classifier",
                    "final_prediction": AgeBracketEnum.UNKNOWN.value,
                    "reason": "trained_weights_missing",
                },
            )
            return AgeResult(
                prediction=AgeBracketEnum.UNKNOWN,
                confidence=0.0,
            )

        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)

        embedding = embedding.to(self.device)

        with torch.no_grad():
            if isinstance(self.model, OrdinalAgeHead):
                probabilities = self.model.predict_probabilities(embedding, temperature=self.temperature).squeeze(0)
            else:
                logits = self.model(embedding)
                scaled_logits = logits / max(self.temperature, 0.1)
                probabilities = F.softmax(scaled_logits, dim=-1).squeeze(0).cpu().numpy()

        max_idx = int(probabilities.argmax())
        calibrated_confidence = float(probabilities[max_idx])
        raw_prediction = self.classes[max_idx]

        # Apply confidence threshold
        if calibrated_confidence < self.confidence_threshold:
            final_prediction = AgeBracketEnum.UNKNOWN
        else:
            final_prediction = raw_prediction

        logger.debug(
            f"Age prediction: {final_prediction.value} (calibrated conf: {calibrated_confidence:.4f}, raw: {raw_prediction.value})",
            extra={
                "event": "age_inference",
                "prediction": final_prediction.value,
                "confidence": calibrated_confidence,
                "temperature": self.temperature,
                "threshold": self.confidence_threshold,
                "probabilities": {cls.value: round(float(p), 4) for cls, p in zip(self.classes, probabilities)},
            },
        )

        return AgeResult(
            prediction=final_prediction,
            confidence=round(calibrated_confidence, 4),
        )


age_classifier = AgeClassifier()
