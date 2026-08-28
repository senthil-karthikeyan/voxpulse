"""Model architectures and classification heads."""

from app.models.gender_classifier import GenderClassifier, GenderClassifierHead
from app.models.age_classifier import AgeClassifier, AgeClassifierHead

__all__ = [
    "GenderClassifier",
    "GenderClassifierHead",
    "AgeClassifier",
    "AgeClassifierHead",
]
