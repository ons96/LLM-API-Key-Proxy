import json
import os
from datetime import datetime
from pathlib import Path
import uuid
from typing import Literal, Dict, Any, Optional
import logging
import time

from .provider_urls import get_provider_endpoint
from .request_database import get_request_database, RequestDatabase


def log_request_to_console(url: str, headers: dict, client_info: tuple, request_data: dict):
    """
    Logs a concise, single-line summary of an incoming request to the console.
    """
    time_str = datetime.now().strftime("%H:%M")
    model_full = request_data.get("model", "N/A")
    
    provider = "N/A"
    model_name = model_full
    endpoint_url = "N/A"

    if '/' in model_full:
        parts = model_full.split('/', 1)
        provider = parts[0]
        model_name = parts[1]
        # Use the helper function to get the full endpoint URL
        endpoint_url = get_provider_endpoint(provider, model_name, url) or "N/A"

    log_message = f"{time_str} - {client_info[0]}:{client_info[1]} - provider: {provider}, model: {model_name} - {endpoint_url}"
    logging.info(log_message)


def log_request_to_database(
    request_id: str,
    client_info: tuple,
    request_data: Dict[str, Any],
    model_full: str,
   ,
) -> Optional endpoint_url: str[Dict[str, Any]]:
    """
    Logs a request to the database.
    
    Args:
        request_id: Unique identifier for the request
        client_info: Tuple of (IP, port)
        request_data: The request payload
        model_full: Full model identifier (e.g., "openai/gpt-4")
        endpoint_url: The full endpoint URL
        
    Returns:
        Dictionary with request metadata for later updating with response
    """
    provider = "N/A"
    model_name = model_full
    
    if '/' in model_full:
        parts = model_full.split('/', 1)
        provider = parts[0]
        model_name = parts[1]
    
    timestamp = datetime.now()
    
    # Get database instance
    db = get_request_database()
    
    # Log the request
    db.log_request(
        request_id=request_id,
        timestamp=timestamp,
        client_ip=client_info[0],
        client_port=client_info[1],
        provider=provider,
        model_name=model_name,
        endpoint_url=endpoint_url,
        request_data=request_data,
        status="pending",
    )
    
    return {
        "request_id": request_id,
        "timestamp": timestamp,
        "provider": provider,
        "model_name": model_name,
    }


def log_response_to_database(
    request_metadata: Dict[str, Any],
    response_data: Optional[Dict[str, Any]] = None,
    error: Optional[Exception] = None,
    duration_ms: Optional[float] = None,
) -> None:
    """
    Updates a request log with response data.
    
    Args:
        request_metadata: Dictionary returned from log_request_to_database
        response_data: The response payload
        error: Exception if request failed
        duration_ms: Request duration in milliseconds
    """
    if not request_metadata or "request_id" not in request_metadata:
        return
    
    db = get_request_database()
    
    if error:
        db.update_request_response(
            request_id=request_metadata["request_id"],
            response_data=None,
            status="error",
            error_message=str(error),
            duration_ms=duration_ms,
        )
    else:
        db.update_request_response(
            request_id=request_metadata["request_id"],
            response_data=response_data,
            status="success",
            error_message=None,
            duration_ms=duration_ms,
        )


class RequestLogger:
    """
    Centralized request logging with both console and database support.
    """
    
    def __init__(self, enable_database: bool = None):
        """
        Initialize the request logger.
        
        Args:
            enable_database: Whether to enable database logging. 
                           Defaults to True if not explicitly disabled.
        """
        if enable_database is None:
            enable_database = os.environ.get("REQUEST_DB_ENABLED", "true").lower() != "false"
        self.enable_database = enable_database
        self._db: Optional[RequestDatabase] = None
        
        if self.enable_database:
            try:
                self._db = get_request_database()
                logging.info("Request database logging enabled")
            except Exception as e:
                logging.warning(f"Failed to initialize request database: {e}")
                self.enable_database = False
    
    def log_request(
        self,
        url: str,
        headers: dict,
        client_info: tuple,
        request_data: dict,
    ) -> Optional[Dict[str, Any]]:
        """
        Log an incoming request to console and optionally database.
        
        Args:
            url: The request URL
            headers: Request headers
            client_info: Tuple of (IP, port)
            request_data: The request payload
            
        Returns:
            Request metadata for tracking response, or None if logging disabled
        """
        # Always log to console
        log_request_to_console(url, headers, client_info, request_data)
        
        # Optionally log to database
        if self.enable_database:
            model_full = request_data.get("model", "N/A")
            
            provider = "N/A"
            model_name = model_full
            endpoint_url = "N/A"

            if '/' in model_full:
                parts = model_full.split('/', 1)
                provider = parts[0]
                model_name = parts[1]
                endpoint_url = get_provider_endpoint(provider, model_name, url) or "N/A"
            
            request_id = str(uuid.uuid4())
            
            return log_request_to_database(
                request_id=request_id,
                client_info=client_info,
                request_data=request_data,
                model_full=model_full,
                endpoint_url=endpoint_url,
            )
        
        return None
    
    def log_response(
        self,
        request_metadata: Optional[Dict[str, Any]],
        response_data: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        start_time: Optional[float] = None,
    ) -> None:
        """
        Log a response to the database.
        
        Args:
            request_metadata: Metadata returned from log_request
            response_data: The response payload
            error: Exception if request failed
            start_time: Optional start time for calculating duration
        """
        if not self.enable_database or not request_metadata:
            return
        
        duration_ms = None
        if start_time is not None:
            duration_ms = (time.time() - start_time) * 1000
        
        log_response_to_database(
            request_metadata=request_metadata,
            response_data=response_data,
            error=error,
            duration_ms=duration_ms,
        )
    
    def get_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list:
        """
        Retrieve request logs from the database.
        
        Args:
            limit: Maximum number of records
            offset: Number of records to skip
            provider: Filter by provider
            model_name: Filter by model name
            status: Filter by status
            
        Returns:
            List of request log dictionaries
        """
        if not self.enable_database or not self._db:
            return []
        
        return self._db.get_request_logs(
            limit=limit,
            offset=offset,
            provider=provider,
            model_name=model_name,
            status=status,
        )
    
    def get_statistics(self, provider: Optional[str] = None) -> list:
        """
        Get request statistics.
        
        Args:
            provider: Optional provider filter
            
        Returns:
            List of statistics dictionaries
        """
        if not self.enable_database or not self._db:
            return []
        
        return self._db.get_statistics(provider=provider)


# Default logger instance
default_logger: Optional[RequestLogger] = None


def get_request_logger() -> RequestLogger:
    """Get or create the default request logger instance."""
    global default_logger
    if default_logger is None:
        default_logger = RequestLogger()
    return default_logger
