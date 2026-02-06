# Configuration Inconsistencies - Fix Required

## Issue: GPT-4 Model Naming Inconsistency

### Current State:
- **Config files** use `gpt-4o` (with number 'o')
- **Codebase expects** `gpt-4` (with letter 'o')

### Problem Locations:
1. `src/rotator_library/score_engine.py:40` - Sets "gpt-4o" as gpt-4o model
2. `src/rotator_library/model_info_service.py:167` - Sets family to "gpt-4o"

### Recommended Fix:
Standardize naming to use **letter 'o'**:

#### Option 1: Update config files (Recommended)
Update all instances of `gpt-4o` to `gpt-4` in:
- `config/virtual_models.yaml` - lines 51, 175, 179
- Any other config files

```bash
# Use sed to replace
sed -i 's/"gpt-4o"/"gpt-4"/g' config/virtual_models.yaml config/router_config.yaml
```

#### Option 2: Update score_engine to accept both forms (Backward Compatible)
Update `src/rotator_library/score_engine.py` line 40 to accept both naming:

```python
# Current (line 40):
"gpt-4o": 68.3,

# Updated:
"gpt-4o": 68.3,
"gpt-4": 68.3,  # Accept both forms
```

### Alternative: Update codebase to use `gpt-4o` everywhere
If codebase uses `gpt-4o` (with letter 'o'), update all references to match config.

### Test After Fix:
After applying fix, verify:
```bash
python -c "from rotator_library.score_engine import GPT_4_SCORE; print(GPT_4_SCORE)"
```

---

## Additional Configuration Inconsistencies to Review:

1. **Free tier models list** - Verify all providers have consistent free_tier_models
2. **Priority ordering** - Ensure numeric priorities work across all providers
3. **Model aliases** - Check for duplicate or conflicting alias definitions

---
*Documented: 2026-02-05*