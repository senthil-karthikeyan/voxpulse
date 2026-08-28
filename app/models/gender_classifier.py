"""Gender classification head for voice attribute inference."""

import os
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.config import settings
from app.core.logging import logger
from app.schemas.response import GenderPredictionEnum, GenderResult


class GenderClassifierHead(nn.Module):
    """PyTorch classification head for predicting gender from voice embeddings."""

    def __init__(self, embedding_dim: int = 192, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.classes = [GenderPredictionEnum.MALE, GenderPredictionEnum.FEMALE]

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, len(self.classes)),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits."""
        return self.classifier(embedding)


class GenderClassifier:
    """Manages gender classification model weights, inference, thresholding, and debug logging."""

    def __init__(
        self,
        embedding_dim: int = 192,
        weights_path: Optional[Path] = None,
        confidence_threshold: float = settings.GENDER_CONFIDENCE_THRESHOLD,
        device: str = settings.DEVICE,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.model = GenderClassifierHead(embedding_dim=embedding_dim).to(self.device)
        self.model.eval()

        self.model_available = False
        self.model_source = "unavailable"
        self.weights_path = weights_path or (settings.MODEL_WEIGHTS_DIR / "gender_head.pt")

        self._load_weights(self.weights_path)

    def _load_weights(self, weights_path: Path) -> None:
        """Load genuine trained model weights if present, or mark classifier as unavailable."""
        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.model_available = True
                self.model_source = "trained"
                logger.info(
                    f"Loaded genuine trained gender classifier weights from {weights_path}",
                    extra={"event": "model_loaded", "model": "gender_classifier", "path": str(weights_path)},
                )
                return
            except Exception as e:
                logger.warning(
                    f"Failed to load trained gender weights from {weights_path}: {e}",
                    extra={"event": "model_load_error", "model": "gender_classifier", "error": str(e)},
                )

        # No trained weights available
        self.model_available = False
        self.model_source = "unavailable"

        if settings.REQUIRE_TRAINED_MODELS:
            raise FileNotFoundError(
                f"Strict mode enabled (REQUIRE_TRAINED_MODELS=True) but trained weights missing at {weights_path}"
            )

        logger.info(
            "Trained gender classifier weights not found. Gender predictions will return 'unknown'.",
            extra={"event": "model_unavailable", "model": "gender_classifier", "reason": "trained_weights_missing"},
        )

    def predict(self, embedding: torch.Tensor) -> GenderResult:
        """Predict gender from voice embedding, log debug metrics, and apply confidence threshold.

        Args:
            embedding: 1D or 2D tensor representing 192-d voice representation.

        Returns:
            GenderResult with prediction ('male', 'female', or 'unknown') and confidence score.
        """
        if not self.model_available:
            logger.debug(
                "Gender inference skipped (model unavailable)",
                extra={
                    "event": "model_unavailable",
                    "model": "gender_classifier",
                    "final_prediction": GenderPredictionEnum.UNKNOWN.value,
                    "reason": "trained_weights_missing",
                },
            )
            return GenderResult(
                prediction=GenderPredictionEnum.UNKNOWN,
                confidence=0.0,
            )

        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)

        embedding = embedding.to(self.device)

        with torch.no_grad():
            logits = self.model(embedding)
            probabilities = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        male_prob = float(probabilities[0])
        female_prob = float(probabilities[1])

        if male_prob >= female_prob:
            raw_prediction = GenderPredictionEnum.MALE
            raw_confidence = male_prob
        else:
            raw_prediction = GenderPredictionEnum.FEMALE
            raw_confidence = female_prob

        # Apply confidence threshold
        if raw_confidence < self.confidence_threshold:
            final_prediction = GenderPredictionEnum.UNKNOWN
        else:
            final_prediction = raw_prediction

        # Structured debug logging
        logger.debug(
            "Gender inference debug",
            extra={
                "event": "gender_debug",
                "model_source": self.model_source,
                "raw_prediction": raw_prediction.value,
                "raw_confidence": round(raw_confidence, 4),
                "threshold": self.confidence_threshold,
                "final_prediction": final_prediction.value,
            },
        )

        return GenderResult(
            prediction=final_prediction,
            confidence=raw_confidence,
        )
