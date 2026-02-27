"""Configuration file validation for YAML configs."""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys

from .data_validator import ValidationResult, validate_providers_database

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration files in the config directory."""
    
    def __init__(self, config_dir: Union[str, Path] = "config"):
        self.config_dir = Path(config_dir)
        self.results: Dict[str, ValidationResult] = {}
    
    def load_yaml(self, filename: str) -> Optional[Dict[str, Any]]:
        """Safely load a YAML configuration file."""
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            logger.error(f"Config file not found: {filepath.absolute()}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                if content is None:
                    logger.warning(f"Empty config file: {filepath}")
                    return {}
                return content
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {filepath}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return None
    
    def validate_providers_database(self) -> ValidationResult:
        """Validate providers_database.yaml."""
        data = self.load_yaml("providers_database.yaml")
        
        if data is None:
            result = ValidationResult()
            result.add_error("Failed to load providers_database.yaml")
            return result
        
        result = validate_providers_database(data)
        self.results['providers_database'] = result
        return result
    
    def validate_model_rankings(self) -> ValidationResult:
        """Validate model_rankings.yaml structure."""
        result = ValidationResult()
        data = self.load_yaml("model_rankings.yaml")
        
        if data is None:
            result.add_error("Failed to load model_rankings.yaml")
            return result
        
        # Basic structure validation
        if not isinstance(data, dict):
            result.add_error("model_rankings.yaml should contain a dictionary")
            return result
        
        if 'rankings' not in data and 'models' not in data:
            result.add_warning("No 'rankings' or 'models' key found in model_rankings.yaml")
        
        result.valid_records = [data] if isinstance(data, dict) else data
        self.results['model_rankings'] = result
        return result
    
    def validate_router_config(self) -> ValidationResult:
        """Validate router_config.yaml."""
        result = ValidationResult()
        data = self.load_yaml("router_config.yaml")
        
        if data is None:
            result.add_error("Failed to load router_config.yaml")
            return result
        
        # Check for required sections
        required_keys = ['routing_strategy', 'fallback_enabled']
        for key in required_keys:
            if key not in data:
                result.add_warning(f"Missing recommended key: {key}")
        
        result.valid_records = [data]
        self.results['router_config'] = result
        return result
    
    def validate_all(self) -> Dict[str, ValidationResult]:
        """Run validation on all configuration files."""
        self.validate_providers_database()
        self.validate_model_rankings()
        self.validate_router_config()
        
        # Add other config files as needed
        # self.validate_virtual_models()
        # self.validate_scoring_config()
        
        return self.results
    
    def is_valid(self, config_name: Optional[str] = None) -> bool:
        """
        Check if configurations are valid.
        
        Args:
            config_name: Specific config to check, or None for all
            
        Returns:
            bool: True if valid
        """
        if config_name:
            result = self.results.get(config_name)
            return result.is_valid if result else False
        
        return all(r.is_valid for r in self.results.values())
    
    def print_report(self):
        """Print validation report to stdout."""
        print("\n" + "="*60)
        print("CONFIGURATION VALIDATION REPORT")
        print("="*60)
        
        for name, result in self.results.items():
            status = "✓ VALID" if result.is_valid else "✗ INVALID"
            print(f"\n{name}: {status}")
            
            if result.errors:
                print("  Errors:")
                for err in result.errors[:5]:  # Show first 5
                    print(f"    - {err}")
                if len(result.errors) > 5:
                    print(f"    ... and {len(result.errors) - 5} more")
            
            if result.warnings:
                print("  Warnings:")
                for warn in result.warnings[:3]:  # Show first 3
                    print(f"    - {warn}")
        
        print("\n" + "="*60)


def main():
    """CLI entry point for config validation."""
    validator = ConfigValidator()
    validator.validate_all()
    validator.print_report()
    
    sys.exit(0 if validator.is_valid() else 1)


if __name__ == "__main__":
    main()
