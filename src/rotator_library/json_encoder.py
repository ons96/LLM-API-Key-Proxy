"""Custom JSON encoder to prevent scientific notation in float serialization."""
import json
import re

class NoScientificNotationEncoder(json.JSONEncoder):
    """JSON encoder that formats floats without scientific notation."""
    
    def encode(self, obj):
        result = super().encode(obj)
        return self._fix_scientific_notation(result)
    
    def _fix_scientific_notation(self, s):
        """Replace scientific notation with decimal format."""
        def replace_float(match):
            num_str = match.group(0)
            try:
                num = float(num_str)
                if num == 0:
                    return "0"
                # Format with enough precision
                if abs(num) < 0.001:
                    formatted = f"{num:.15f}"
                else:
                    formatted = f"{num:.10f}"
                # Strip trailing zeros
                if "." in formatted:
                    formatted = formatted.rstrip("0").rstrip(".")
                return formatted
            except:
                return num_str
        
        # Match floats in scientific notation (with or without decimal point before e)
        return re.sub(r"-?\d+(?:\.\d+)?[eE][+-]\d+", replace_float, s)

def json_dumps_no_scientific(obj):
    """Convenience function to serialize without scientific notation."""
    return json.dumps(obj, cls=NoScientificNotationEncoder)
