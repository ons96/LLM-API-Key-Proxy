"""
Routing feature extraction for smart routing (#476/#477/#478/#480).

Exports the request feature extractor used by the dynamic-chain middleware.
"""

from .request_features import (
    Capabilities,
    PromptBucket,
    RequestFeatures,
    TaskClass,
    extract_request_features,
)
from .output_estimator import (
    DEFAULT_OUTPUT_TOKENS,
    OutputEstimator,
)
from .tier_classifier import (
    TIER_FLOORS,
    TASK_TIER,
    Tier,
    classify_request,
    load_model_scores,
    model_meets_floor,
    parse_tier_header,
)
from .latency_predictor import (
    LatencyPredictor,
    Prediction,
)

__all__ = [
    "Capabilities",
    "PromptBucket",
    "RequestFeatures",
    "TaskClass",
    "extract_request_features",
    "DEFAULT_OUTPUT_TOKENS",
    "OutputEstimator",
    "TIER_FLOORS",
    "TASK_TIER",
    "Tier",
    "classify_request",
    "load_model_scores",
    "model_meets_floor",
    "parse_tier_header",
    "LatencyPredictor",
    "Prediction",
]
