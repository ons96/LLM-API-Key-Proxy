from sqlalchemy import Column, String, Float, DateTime, JSON, Integer, create_engine, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class ArtificialAnalysisModel(Base):
    __tablename__ = "artificial_analysis_models"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    slug = Column(String, nullable=False, index=True)
    model_creator_name = Column(String, index=True)
    model_creator_slug = Column(String)
    median_output_tokens_per_second = Column(Float)
    
    # Store dynamic evaluations and pricing as JSON for flexibility
    evaluations = Column(JSON)
    pricing = Column(JSON)
    
    # Extracted key metrics for indexing and sorting
    intelligence_index = Column(Float, index=True)
    
    # Timestamps
    fetched_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Composite index for common queries
    __table_args__ = (
        Index('idx_creator_intelligence', 'model_creator_name', 'intelligence_index'),
    )

class DataFetchLog(Base):
    __tablename__ = "data_fetch_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(100), nullable=False, index=True)  # e.g., 'artificial_analysis'
    status = Column(String(20), nullable=False)  # 'success', 'failure'
    records_count = Column(Integer, default=0)
    error_message = Column(String(500), nullable=True)
    fetched_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
