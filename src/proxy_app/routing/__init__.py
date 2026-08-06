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

__all__ = [
    "Capabilities",
    "PromptBucket",
    "RequestFeatures",
    "TaskClass",
    "extract_request_features",
]
