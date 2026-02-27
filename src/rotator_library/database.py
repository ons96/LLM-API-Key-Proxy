import os
import yaml
from typing import Optional, Generator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool

from .db_models import Base

class DatabaseManager:
    def __init__(self, config_path: str = "config/database.yaml"):
        self.config = self._load_config(config_path)
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def _load_config(self, path: str) -> dict:
        """Load database configuration from YAML."""
        if not os.path.exists(path):
            return {"database": {"url": "sqlite:///data/rotator_library.db"}}
        
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_engine(self):
        """Create SQLAlchemy engine with appropriate configuration for SQLite or PostgreSQL."""
        db_url = os.getenv(
            "DATABASE_URL", 
            self.config.get("database", {}).get("url", "sqlite:///data/rotator_library.db")
        )
        
        # Ensure data directory exists for SQLite
        if db_url.startswith("sqlite"):
            db_path = db_url.replace("sqlite:///", "")
            if db_path != ":memory:":
                os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            
            sqlite_config = self.config.get("database", {}).get("sqlite", {})
            connect_args = {}
            if sqlite_config.get("check_same_thread", False) == False:
                connect_args["check_same_thread"] = False
            
            # Use StaticPool for SQLite to handle threading properly
            return create_engine(
                db_url,
                connect_args=connect_args,
                poolclass=StaticPool,
                echo=False
            )
        else:
            # PostgreSQL configuration with connection pooling
            pool_size = int(os.getenv(
                "DB_POOL_SIZE", 
                self.config.get("database", {}).get("pool_size", 5)
            ))
            max_overflow = int(os.getenv(
                "DB_MAX_OVERFLOW", 
                self.config.get("database", {}).get("max_overflow", 10)
            ))
            
            engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=self.config.get("database", {}).get("pool_timeout", 30),
                pool_recycle=self.config.get("database", {}).get("pool_recycle", 1800),
                echo=False
            )
            
            # Enable SSL if configured
            ssl_mode = self.config.get("database", {}).get("postgresql", {}).get("ssl_mode")
            if ssl_mode and ssl_mode != "disable":
                # SSL is handled in the connection URL or connect_args for PostgreSQL
                pass
            
            return engine
    
    def create_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all tables. Use with caution."""
        Base.metadata.drop_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def close(self):
        """Dispose the engine and close connections."""
        self.engine.dispose()

# Global singleton instance
_db_manager: Optional[DatabaseManager] = None

def get_db_manager() -> DatabaseManager:
    """Get or create the global DatabaseManager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency generator for database sessions."""
    db = get_db_manager().get_session()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database tables. Call this on application startup."""
    manager = get_db_manager()
    manager.create_tables()
    return manager
