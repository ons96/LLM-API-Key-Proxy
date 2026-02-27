"""Data validation utilities for the rotator library."""

import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from pydantic import ValidationError
import pandas as pd

from .data_schemas import (
    ArtificialAnalysisModel, 
    ProvidersDatabase, 
    Provider,
    DataValidationReport
)

logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.valid_records: List[Dict[str, Any]] = []
        self.invalid_records: List[Tuple[Dict[str, Any], str]] = []
    
    def add_error(self, message: str, record: Optional[Dict] = None):
        self.is_valid = False
        self.errors.append(message)
        if record is not None:
            self.invalid_records.append((record, message))
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def add_valid_record(self, record: Dict[str, Any]):
        self.valid_records.append(record)
    
    def to_report(self) -> DataValidationReport:
        """Convert to serializable report."""
        return DataValidationReport(
            record_count=len(self.valid_records) + len(self.invalid_records),
            valid_count=len(self.valid_records),
            error_count=len(self.errors),
            warning_count=len(self.warnings),
            errors=self.errors,
            warnings=self.warnings,
            is_valid=self.is_valid and len(self.valid_records) > 0
        )


def validate_artificial_analysis_record(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a single record against the ArtificialAnalysisModel schema."""
    try:
        ArtificialAnalysisModel(**record)
        return True, None
    except ValidationError as e:
        error_msg = "; ".join([f"{err['loc']}: {err['msg']}" for err in e.errors()])
        return False, error_msg


def validate_artificial_analysis_data(
    data: List[Dict[str, Any]], 
    strict: bool = False,
    allow_partial: bool = True
) -> ValidationResult:
    """
    Validate a list of processed records from Artificial Analysis API.
    
    Args:
        data: List of model records
        strict: If True, treat warnings as errors
        allow_partial: If True, return valid records even if some fail
        
    Returns:
        ValidationResult with details of validation
    """
    result = ValidationResult()
    
    if not data:
        result.add_error("Empty data set provided")
        return result
    
    if not isinstance(data, list):
        result.add_error(f"Expected list of records, got {type(data)}")
        return result
    
    required_fields = {'id', 'name', 'slug'}
    
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            result.add_error(f"Record {idx}: Expected dict, got {type(record)}")
            continue
            
        # Check required fields
        missing_fields = required_fields - set(record.keys())
        if missing_fields:
            result.add_error(f"Record {idx}: Missing required fields {missing_fields}", record)
            if not allow_partial:
                continue
        
        # Schema validation
        is_valid, error_msg = validate_artificial_analysis_record(record)
        if is_valid:
            result.add_valid_record(record)
        else:
            result.add_error(f"Record {idx} (ID: {record.get('id', 'unknown')}): {error_msg}", record)
            if not allow_partial:
                continue
        
        # Data quality checks
        intel_index = record.get('eval_artificial_analysis_intelligence_index')
        if intel_index is not None:
            try:
                val = float(intel_index)
                if not 0 <= val <= 100:
                    msg = f"Record {idx}: Intelligence index out of range (0-100): {val}"
                    if strict:
                        result.add_error(msg, record)
                    else:
                        result.add_warning(msg)
            except (ValueError, TypeError):
                msg = f"Record {idx}: Non-numeric intelligence index: {intel_index}"
                if strict:
                    result.add_error(msg, record)
                else:
                    result.add_warning(msg)
        
        # Check for negative throughput
        throughput = record.get('median_output_tokens_per_second')
        if throughput is not None and throughput < 0:
            msg = f"Record {idx}: Negative throughput value: {throughput}"
            if strict:
                result.add_error(msg, record)
            else:
                result.add_warning(msg)
    
    logger.info(
        f"Validation complete: {len(result.valid_records)} valid, "
        f"{len(result.errors)} errors, {len(result.warnings)} warnings"
    )
    return result


def validate_providers_database(data: Dict[str, Any]) -> ValidationResult:
    """Validate providers_database.yaml structure."""
    result = ValidationResult()
    
    try:
        validated = ProvidersDatabase(**data)
        
        # Convert back to dict for consistency
        for provider in validated.providers:
            result.add_valid_record(provider.dict())
            
        logger.info(f"Validated {len(validated.providers)} providers")
        
    except ValidationError as e:
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error['loc'])
            result.add_error(f"{loc}: {error['msg']}")
    except Exception as e:
        result.add_error(f"Unexpected validation error: {e}")
    
    return result


def validate_csv_dataframe(df: pd.DataFrame, strict: bool = False) -> ValidationResult:
    """
    Validate a pandas DataFrame before saving to CSV.
    
    Args:
        df: DataFrame to validate
        strict: If True, apply stricter validation rules
        
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    if df.empty:
        result.add_error("DataFrame is empty")
        return result
    
    # Required columns check
    required_columns = {'id', 'name', 'slug'}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        result.add_error(f"Missing required columns: {missing_cols}")
    
    # Duplicate ID check
    if 'id' in df.columns:
        duplicates = df[df.duplicated(subset=['id'], keep=False)]
        if not duplicates.empty:
            dup_ids = duplicates['id'].tolist()
            result.add_error(f"Duplicate IDs found: {dup_ids}")
    
    # Null checks in required fields
    for col in required_columns & set(df.columns):
        null_count = df[col].isnull().sum()
        if null_count > 0:
            if strict:
                result.add_error(f"Column '{col}' has {null_count} null values")
            else:
                result.add_warning(f"Column '{col}' has {null_count} null values")
    
    # Data type checks
    if 'median_output_tokens_per_second' in df.columns:
        non_numeric = df['median_output_tokens_per_second'].apply(
            lambda x: not isinstance(x, (int, float, type(None)))
        ).sum()
        if non_numeric > 0:
            result.add_warning(f"{non_numeric} records have non-numeric throughput values")
    
    if result.is_valid:
        result.valid_records = df.to_dict('records')
    
    return result


def validate_data_freshness(
    timestamp_file: str, 
    max_age_hours: int = 24
) -> Tuple[bool, Optional[str]]:
    """
    Validate that cached data is not too old.
    
    Returns:
        Tuple of (is_fresh, error_message)
    """
    from datetime import datetime, timedelta
    import os
    
    if not os.path.exists(timestamp_file):
        return False, "No timestamp file found"
    
    try:
        with open(timestamp_file, 'r') as f:
            last_fetch = datetime.fromisoformat(f.read().strip())
        
        age = datetime.now() - last_fetch
        if age > timedelta(hours=max_age_hours):
            return False, f"Data is {age.total_seconds() / 3600:.1f} hours old (max {max_age_hours})"
        
        return True, None
        
    except Exception as e:
        return False, f"Error reading timestamp: {e}"
