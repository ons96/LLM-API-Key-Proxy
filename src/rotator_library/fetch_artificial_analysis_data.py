import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='c:/Users/owens/Coding Projects/.env')

# --- Configuration ---
API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
# API key is loaded from .env file
API_KEY = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY") 
FIXED_OUTPUT_FILENAME = "artificial_analysis_models.csv" # Fixed output CSV filename
TIMESTAMP_FILE = "last_successful_fetch.txt" # Stores timestamp of last successful fetch
CACHE_DURATION_HOURS = 24 # Cache duration in hours

# Fields to extract from the API response as requested by the user
# We will flatten nested objects like 'model_creator', 'evaluations', and 'pricing'
REQUESTED_FIELDS = [
    "id",
    "name",
    "slug",
    "model_creator.name", # Flattened from model_creator object
    "model_creator.slug", # Flattened from model_creator object
    "evaluations",        # This will be a dict, further processing might be needed depending on desired CSV structure
    "pricing",            # This will also be a dict
    "median_output_tokens_per_second"
]

# Fields to explicitly exclude as per user request (though the API call fetches all, we just won't save these)
EXCLUDED_FIELDS = [
    "median_time_to_first_token_seconds",
    "median_time_to_first_answer_token"
]

def is_cache_valid(timestamp_file, duration_hours):
    """Checks if the cache timestamp is valid and within the specified duration."""
    if not os.path.exists(timestamp_file):
        return False
    try:
        with open(timestamp_file, 'r') as f:
            last_fetch_time_str = f.read().strip()
        last_fetch_time = datetime.fromisoformat(last_fetch_time_str)
        if datetime.now() - last_fetch_time < timedelta(hours=duration_hours):
            return True
    except Exception as e:
        print(f"Error reading or parsing timestamp file {timestamp_file}: {e}")
        # If timestamp is invalid, treat cache as invalid
        return False
    return False

def update_success_timestamp(timestamp_file):
    """Updates the timestamp file with the current datetime."""
    try:
        with open(timestamp_file, 'w') as f:
            f.write(datetime.now().isoformat())
        print(f"Successfully updated timestamp in {timestamp_file}")
    except Exception as e:
        print(f"Error updating timestamp file {timestamp_file}: {e}")

def fetch_data_from_api(api_key):
    """Fetches all model data from the Artificial Analysis API."""
    if not api_key:
        print("ERROR: ARTIFICIAL_ANALYSIS_API_KEY not found in .env file or is not set.")
        return None

    headers = {
        "x-api-key": api_key
    }
    print(f"Fetching data from {API_URL}...")
    try:
        response = requests.get(API_URL, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        print("Data fetched successfully.")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {response.status_code} {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except json.JSONDecodeError as json_err:
        print(f"Failed to decode JSON response: {json_err}")
    return None

def process_api_data(api_response_data):
    """Processes the API response to extract required fields and flatten nested structures."""
    if not api_response_data or 'data' not in api_response_data:
        print("No data found in API response or response is malformed.")
        return []

    all_models_data = api_response_data['data']
    processed_records = []

    for model_data in all_models_data:
        record = {}
        record['id'] = model_data.get('id')
        record['name'] = model_data.get('name')
        record['slug'] = model_data.get('slug')
        
        # Flatten model_creator
        model_creator = model_data.get('model_creator', {})
        record['model_creator_name'] = model_creator.get('name')
        record['model_creator_slug'] = model_creator.get('slug')
        
        # Flatten evaluations
        evaluations_data = model_data.get('evaluations', {})
        if isinstance(evaluations_data, dict):
            for eval_key, eval_value in evaluations_data.items():
                # Prefix to avoid collision with other keys and identify source
                record[f'eval_{eval_key}'] = eval_value
        elif evaluations_data is not None: # If it's not a dict but not None (e.g. list, string)
            record['evaluations_as_json'] = json.dumps(evaluations_data) # Store as JSON string under a new key
        
        # Flatten pricing
        pricing_data = model_data.get('pricing', {})
        if isinstance(pricing_data, dict):
            for price_key, price_value in pricing_data.items():
                record[f'price_{price_key}'] = price_value
        elif pricing_data is not None: # If it's not a dict but not None
            record['pricing_as_json'] = json.dumps(pricing_data) # Store as JSON string under a new key
        
        record['median_output_tokens_per_second'] = model_data.get('median_output_tokens_per_second')
        
        # Add other top-level fields if they were directly requested and not nested
        # (Example: if 'release_date' was needed, add: record['release_date'] = model_data.get('release_date'))

        processed_records.append(record)
    
    print(f"Processed {len(processed_records)} model records.")
    return processed_records

def save_data_to_csv(data, filename):
    """Saves the processed data to a CSV file. Returns True on success, False on failure."""
    if not data:
        print("No data to save.")
        return False # Indicate failure or no action

    df = pd.DataFrame(data)
    try:
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Data successfully saved to {filename}")
        return True # Indicate success
    except Exception as e:
        print(f"Error saving data to CSV: {e}")
        return False # Indicate failure

if __name__ == "__main__":
    print("Starting Artificial Analysis API data fetcher...")

    output_csv_path = FIXED_OUTPUT_FILENAME
    timestamp_file_path = TIMESTAMP_FILE

    if is_cache_valid(timestamp_file_path, CACHE_DURATION_HOURS):
        print(f"Data fetched within the last {CACHE_DURATION_HOURS} hours. Using cached version from '{output_csv_path}'. Script will not fetch new data.")
    else:
        print("Cache is invalid or expired. Attempting to fetch new data...")
        api_response = fetch_data_from_api(API_KEY)
        
        if api_response: # API call was successful at a basic level
            processed_data = process_api_data(api_response)
            if processed_data: # Actual data was extracted and processed
                print(f"Successfully fetched and processed {len(processed_data)} records.")
                
                # Sort data by 'eval_artificial_analysis_intelligence_index' descending
                # Handles None or missing values by treating them as -infinity (sorting to the bottom)
                def get_sort_value(item):
                    value = item.get('eval_artificial_analysis_intelligence_index')
                    if value is None: 
                        return -float('inf') 
                    try:
                        return float(value) 
                    except (ValueError, TypeError):
                        return -float('inf')
                
                processed_data.sort(key=get_sort_value, reverse=True)
                print("Data sorted by 'eval_artificial_analysis_intelligence_index' (descending).")

                print("Attempting to save sorted data...")
                save_successful = save_data_to_csv(processed_data, output_csv_path)
                
                if save_successful:
                    update_success_timestamp(timestamp_file_path)
                    print(f"Successfully fetched, processed, sorted, saved data to '{output_csv_path}', and updated timestamp.")
                else:
                    print(f"Failed to save data to '{output_csv_path}'. Timestamp not updated.")
            else:
                print(f"API call successful, but no new model data was processed. Existing CSV file '{output_csv_path}' (if any) will not be overwritten. Timestamp not updated.")
        else:
            print(f"Failed to fetch data from API. Existing CSV file '{output_csv_path}' (if any) will not be overwritten. Timestamp not updated.")
    
    print("Script finished.")