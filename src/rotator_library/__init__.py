import logging

# Configure the main library logger
logger = logging.getLogger("rotator_library")
logger.setLevel(logging.INFO)

# Add NullHandler to prevent "no handlers" warning
logger.addHandler(logging.NullHandler())

# Import public API
from .failure_logger import (
    configure_failure_logger,
    get_failure_logger,
    log_failure,
    main_lib_logger,
)
from .error_handler import (
    NoAvailableKeysError,
    PreRequestCallbackError,
    CredentialNeedsReauthError,
    EmptyResponseError,
    ABNORMAL_ERROR_TYPES,
    NORMAL_ERROR_TYPES,
    is_abnormal_error,
    extract_retry_after_from_body,
)
from .client import Client
from .provider_factory import ProviderFactory
from .model_definitions import ModelDefinitions
from .model_info_service import ModelInfoService
from .utils.paths import get_logs_dir, get_cache_dir, get_temp_dir, get_config_dir
