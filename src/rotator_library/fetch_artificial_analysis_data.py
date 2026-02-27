import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = os.getenv('ENV_PATH', 'c:/Users/owens/Coding Projects/.env')
load_dotenv(dotenv_path=env_path)

# --- Configuration ---
API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
API_KEY = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY")
FIXED_OUTPUT_FILENAME = "artificial_analysis_models.csv"
TIMESTAMP_FILE = "last_successful_fetch.txt"
CACHE_DURATION_HOURS = 24

# Fields to extract from the API response
REQUESTED_FIELDS = [
    "id",
    "name",
    "slug",
    "model_creator.name",
    "model_creator.slug",
    "evaluations",
    "pricing",
    "median_output_tokens_per_second"
]

EXCLUDED_FIELDS = [
    "median_time_to_first_token_seconds",
    "median_time_to_first_answer_token"
]


class DataCollectionError(Exception):
    """Base exception for data collection errors."""
    pass


class APIAuthenticationError(DataCollectionError):
    """Raised when API authentication fails."""
    pass


class APIRequestError(DataCollectionError):
    """Raised when API request fails."""
    pass


def is_cache_valid(timestamp_file: str, duration_hours: int) -> bool:
    """
    Check if cached data is still valid based on timestamp.
    
    Args:
        timestamp_file: Path to file containing last fetch timestamp
        duration_hours: Cache validity duration in hours
        
    Returns:
        True if cache is valid, False otherwise
    """
    if not os.path.exists(timestamp_file):
        return False
    
    try:
        with open(timestamp_file, 'r', encoding='utf-8') as f:
            last_fetch_time_str = f.read().strip()
        last_fetch_time = datetime.fromisoformat(last_fetch_time_str)
        
        if datetime.now() - last_fetch_time < timedelta(hours=duration_hours):
            logger.info(f"Cache valid. Last fetch: {last_fetch_time.isoformat()}")
            return True
            
    except (ValueError, IOError) as e:
        logger.warning(f"Error reading timestamp file: {e}")
        
    return False


def update_success_timestamp(timestamp_file: str) -> None:
    """Update timestamp file with current datetime."""
    try:
        with open(timestamp_file, 'w', encoding='utf-8') as f:
            f.write(datetime.now().isoformat())
        logger.info(f"Updated timestamp in {timestamp_file}")
    except IOError as e:
        logger.error(f"Failed to update timestamp: {e}")
        raise DataCollectionError(f"Cannot write timestamp: {e}")


def fetch_data_from_api(api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Fetch model data from Artificial Analysis API.
    
    Args:
        api_key: API authentication key
        
    Returns:
        Parsed JSON response or None if request failed
        
    Raises:
        APIAuthenticationError: If API key is missing or invalid
        APIRequestError: If HTTP request fails
    """
    if not api_key:
        logger.error("ARTIFICIAL_ANALYSIS_API_KEY not found in environment")
        raise APIAuthenticationError("API key not configured")
    
    headers = {"x-api-key": api_key}
    logger.info(f"Fetching data from {API_URL}")
    
    try:
        response = requests.get(API_URL, headers=headers, timeout=30)
        response.raise_for_status()
        logger.info("Successfully fetched data from API")
        return response.json()
        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            raise APIAuthenticationError("Invalid API key") from e
        raise APIRequestError(f"HTTP {response.status_code}: {e}") from e
    except requests.exceptions.Timeout:
        raise APIRequestError("Request timeout after 30s")
    except requests.exceptions.RequestException as e:
        raise APIRequestError(f"Request failed: {e}") from e


def process_api_data(api_response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process API response into flattened records.
    
    Transforms nested structures (evaluations, pricing, model_creator)
    into flat key-value pairs suitable for CSV export.
    
    Args:
        api_response_data: Raw API response JSON
        
    Returns:
        List of processed model records
    """
    if not api_response_data or 'data' not in api_response_data:
        logger.warning("No data found in API response")
        return []
    
    all_models_data = api_response_data['data']
    processed_records = []
    
    for model_data in all_models_data:
        record = {
            'id': model_data.get('id'),
            'name': model_data.get('name'),
            'slug': model_data.get('slug'),
            'median_output_tokens_per_second': model_data.get('median_output_tokens_per_second')
        }
        
        # Flatten model_creator
        model_creator = model_data.get('model_creator', {})
        record['model_creator_name'] = model_creator.get('name')
        record['model_creator_slug'] = model_creator.get('slug')
        
        # Flatten evaluations with prefix
        evaluations = model_data.get('evaluations', {})
        if isinstance(evaluations, dict):
            for key, value in evaluations.items():
                record[f'eval_{key}'] = value
        elif evaluations is not None:
            record['evaluations_raw'] = json.dumps(evaluations)
        
        # Flatten pricing with prefix
        pricing = model_data.get('pricing', {})
        if isinstance(pricing, dict):
            for key, value in pricing.items():
                record[f'price_{key}'] = value
        elif pricing is not None:
            record['pricing_raw'] = json.dumps(pricing)
        
        processed_records.append(record)
    
    logger.info(f"Processed {len(processed_records)} model records")
    return processed_records


def sort_by_intelligence_index(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort records by Artificial Analysis Intelligence Index (descending).
    
    Handles None/invalid values by sorting them to bottom.
    """
    def get_sort_value(item: Dict[str, Any]) -> float:
        value = item.get('eval_artificial_analysis_intelligence_index')
        if value is None:
            return -float('inf')
        try:
            return float(value)
        except (ValueError, TypeError):
            return -float('inf')
    
    records.sort(key=get_sort_value, reverse=True)
    logger.info("Sorted records by intelligence index")
    return records


def save_data_to_csv(data: List[Dict[str, Any]], filename: str) -> bool:
    """
    Save processed data to CSV file.
    
    Args:
        data: List of model records
        filename: Output CSV path
        
    Returns:
        True if successful, False otherwise
    """
    if not data:
        logger.warning("No data to save")
        return False
    
    try:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"Saved {len(data)} records to {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        return False


def run_collection_pipeline(
    output_path: str = FIXED_OUTPUT_FILENAME,
    timestamp_path: str = TIMESTAMP_FILE,
    force_refresh: bool = False
) -> bool:
    """
    Execute full data collection pipeline.
    
    Args:
        output_path: Path for output CSV
        timestamp_path: Path for timestamp file
        force_refresh: Ignore cache and force new fetch
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Starting data collection pipeline")
    
    if not force_refresh and is_cache_valid(timestamp_path, CACHE_DURATION_HOURS):
        logger.info("Using cached data")
        return True
    
    try:
        api_response = fetch_data_from_api(API_KEY)
        if not api_response:
            return False
        
        processed_data = process_api_data(api_response)
        if not processed_data:
            logger.warning("No data processed from API response")
            return False
        
        sorted_data = sort_by_intelligence_index(processed_data)
        
        if save_data_to_csv(sorted_data, output_path):
            update_success_timestamp(timestamp_path)
            logger.info("Pipeline completed successfully")
            return True
        return False
        
    except APIAuthenticationError as e:
        logger.error(f"Authentication failed: {e}")
        return False
    except APIRequestError as e:
        logger.error(f"API request failed: {e}")
        return False
    except Exception as e:
        logger.exception("Unexpected error in pipeline")
        return False


if __name__ == "__main__":
    success = run_collection_pipeline()
    exit(0 if success else 1)
