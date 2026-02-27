"""
Request Database Module

Handles persistent storage of request logs to a SQLite database.
"""

import sqlite3
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Default database path - can be overridden via environment variable
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "requests.db")


class RequestDatabase:
    """
    Manages SQLite database for request logging.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("REQUEST_DB_PATH", DEFAULT_DB_PATH)
        self._ensure_db_directory()
        self._init_database()

    def _ensure_db_directory(self):
        """Create database directory if it doesn't exist."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            Path(db_dir).mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL UNIQUE,
                    timestamp TEXT NOT NULL,
                    client_ip TEXT,
                    client_port INTEGER,
                    provider TEXT,
                    model_name TEXT,
                    endpoint_url TEXT,
                    request_data TEXT,
                    response_data TEXT,
                    status TEXT,
                    error_message TEXT,
                    duration_ms REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for common query patterns
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_request_logs_timestamp 
                ON request_logs(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_request_logs_provider 
                ON request_logs(provider)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_request_logs_model 
                ON request_logs(model_name)
            """)
            
            conn.commit()
            logger.info(f"Request database initialized at {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def log_request(
        self,
        request_id: str,
        timestamp: datetime,
        client_ip: str,
        client_port: int,
        provider: str,
        model_name: str,
        endpoint_url: str,
        request_data: Dict[str, Any],
        response_data: Optional[Dict[str, Any]] = None,
        status: str = "pending",
        error_message: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """
        Log a request to the database.
        
        Args:
            request_id: Unique identifier for the request
            timestamp: Request timestamp
            client_ip: Client IP address
            client_port: Client port
            provider: Provider name
            model_name: Model name
            endpoint_url: Full endpoint URL
            request_data: The request payload (will be JSON serialized)
            response_data: The response payload (will be JSON serialized)
            status: Request status (pending, success, error)
            error_message: Error message if status is error
            duration_ms: Request duration in milliseconds
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO request_logs (
                        request_id, timestamp, client_ip, client_port,
                        provider, model_name, endpoint_url, request_data,
                        response_data, status, error_message, duration_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                        client_ip,
                        client_port,
                        provider,
                        model_name,
                        endpoint_url,
                        json.dumps(request_data),
                        json.dumps(response_data) if response_data else None,
                        status,
                        error_message,
                        duration_ms,
                    ),
                )
                conn.commit()
                logger.debug(f"Request {request_id} logged to database")
        except Exception as e:
            logger.error(f"Failed to log request to database: {e}")

    def update_request_response(
        self,
        request_id: str,
        response_data: Dict[str, Any],
        status: str = "success",
        error_message: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """
        Update a pending request with its response data.
        
        Args:
            request_id: The request ID to update
            response_data: The response payload
            status: Final status (success, error)
            error_message: Error message if status is error
            duration_ms: Request duration in milliseconds
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE request_logs 
                    SET response_data = ?, status = ?, error_message = ?, duration_ms = ?
                    WHERE request_id = ?
                    """,
                    (
                        json.dumps(response_data),
                        status,
                        error_message,
                        duration_ms,
                        request_id,
                    ),
                )
                conn.commit()
                logger.debug(f"Request {request_id} response logged to database")
        except Exception as e:
            logger.error(f"Failed to update request in database: {e}")

    def get_request_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve request logs with optional filtering.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            provider: Filter by provider
            model_name: Filter by model name
            status: Filter by status
            
        Returns:
            List of request log dictionaries
        """
        query = "SELECT * FROM request_logs WHERE 1=1"
        params = []
        
        if provider:
            query += " AND provider = ?"
            params.append(provider)
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        if status:
            query += " AND status = ?"
            params.append(status)
            
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [
                    {
                        "request_id": row["request_id"],
                        "timestamp": row["timestamp"],
                        "client_ip": row["client_ip"],
                        "client_port": row["client_port"],
                        "provider": row["provider"],
                        "model_name": row["model_name"],
                        "endpoint_url": row["endpoint_url"],
                        "request_data": json.loads(row["request_data"]) if row["request_data"] else None,
                        "response_data": json.loads(row["response_data"]) if row["response_data"] else None,
                        "status": row["status"],
                        "error_message": row["error_message"],
                        "duration_ms": row["duration_ms"],
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to retrieve request logs: {e}")
            return []

    def get_statistics(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get request statistics.
        
        Args:
            provider: Optional provider filter
            
        Returns:
            Dictionary with statistics
        """
        query = """
            SELECT 
                COUNT(*) as total_requests,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_requests,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as failed_requests,
                AVG(duration_ms) as avg_duration_ms,
                provider,
                model_name
            FROM request_logs
        """
        params = []
        
        if provider:
            query += " WHERE provider = ?"
            params.append(provider)
            
        query += " GROUP BY provider, model_name"
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [
                    {
                        "provider": row["provider"],
                        "model_name": row["model_name"],
                        "total_requests": row["total_requests"],
                        "successful_requests": row["successful_requests"],
                        "failed_requests": row["failed_requests"],
                        "avg_duration_ms": row["avg_duration_ms"],
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to retrieve statistics: {e}")
            return []


# Global instance
_request_db: Optional[RequestDatabase] = None


def get_request_database() -> RequestDatabase:
    """Get or create the global request database instance."""
    global _request_db
    if _request_db is None:
        _request_db = RequestDatabase()
    return _request_db
