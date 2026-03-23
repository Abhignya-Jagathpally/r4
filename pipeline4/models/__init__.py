"""Model implementations."""
from .cox_ph import CoxPHModel
from .deepsurv import DeepSurvModel
from .rsf import RSFModel
from .treatment_classifier import TreatmentResponseClassifier
from .attention_fusion import MultiModalAttentionFusion

MODEL_REGISTRY = {
    "cox_ph": CoxPHModel,
    "deepsurv": DeepSurvModel,
    "rsf": RSFModel,
    "response_classifier": TreatmentResponseClassifier,
    "attention_fusion": MultiModalAttentionFusion,
}

__all__ = [
    "CoxPHModel", "DeepSurvModel", "RSFModel",
    "TreatmentResponseClassifier", "MultiModalAttentionFusion",
    "MODEL_REGISTRY",
]
