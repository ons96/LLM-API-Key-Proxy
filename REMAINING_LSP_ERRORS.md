# Remaining LSP Errors (Pre-existing Issues)

**Documented:** 2026-02-07
**Note:** These errors existed before type hint improvements and were not introduced by recent changes.

---

## 🚨 High Priority Issues

### 1. config/router_config.yaml (YAML Formatting)
**Error:** "All sequence items must start at same column"
**Location:** Line 157, Column 6
**Impact:** YAML configuration file has indentation inconsistency
**Fix:** Review router_config.yaml around line 157 and ensure consistent indentation

### 2. src/proxy_app/detailed_logger.py (Wrong Type Passed)
**Errors:**
- Line 42: `logging` module passed instead of `logger` instance
- Line 48: `logging` module passed instead of `logger` instance
- Line 55: `logging` module passed instead of `logger` instance
- Line 79: `logging` module passed instead of `logger` instance

**Impact:** Functions receive wrong type parameter
**Fix:** Replace `logging` with `logger` (line 14: `logger = logging.getLogger(__name__)`)

---

## ⚠️ Medium Priority Issues

### 3. src/proxy_app/main.py (Type Mismatches)
**Errors:**
- Line 416: `_Environ[str]` not assignable to `Dict[str, str]`
- Line 508: `BaseException` not iterable (when unpacking)
- Line 601: `set_verbose` not exported from `litellm`
- Line 826: `dict[str, str]` not assignable to `str`
- Line 832: String subscript with literal keys not supported
- Line 837: String subscript with literal keys not supported
- Line 860: `list[Unknown]` not assignable to `str`
- Line 868: `None` not assignable to `str`
- Line 941: `bytes | memoryview[int]` not assignable to `str | bytes | bytearray`
- Lines 1080, 1082, 1085, 1087, 1089, 1091, 1093, 1095: Exception types not exported from `litellm`

**Impact:** Type safety issues, potential runtime errors
**Fix Required:**
1. Fix environment variable handling
2. Use proper exception types from `litellm`
3. Fix string subscript operations (use dict instead)
4. Fix type assignments

### 4. src/proxy_app/router_core.py (Type Mismatches)
**Errors:**
- Line 186: `None` not assignable to `List[str]`
- Line 271: `None` not assignable to `List[str]`
- Line 600: Complex union return type not assignable to `Dict[str, Any]`
- Line 979: `None` not assignable to `List[Dict[Unknown, Unknown]]`
- Line 1060: `None` not assignable to `List[Dict[Unknown, Unknown]]`
- Line 1283: `ModelResponse` not iterable (missing `__aiter__`)

**Impact:** Type safety issues, potential runtime errors
**Fix Required:**
1. Return empty list `[]` instead of `None` where `List[str]` expected
2. Fix return type of chat_completions function
3. Fix iterable handling for ModelResponse

---

## 📋 Low Priority Issues

### 5. src/proxy_app/health_checker.py (Pre-existing)
**Error:** Line 95: `BaseException` not iterable
**Note:** This error is expected behavior when using `return_exceptions=True` with `asyncio.gather`

**Context:**
```python
results = await asyncio.gather(*tasks, return_exceptions=True)
for result in results:
    if isinstance(result, Exception):
        # This is intentional error handling
```

**Impact:** False positive from LSP - code is correct
**Fix:** None required (code handles exceptions as expected)

---

## 🎯 Recommended Action Plan

### Phase 1: Critical Fixes (Do Immediately)
1. **Fix config/router_config.yaml** indentation
2. **Fix detailed_logger.py** - replace `logging` module with `logger` instance

### Phase 2: Type Safety Fixes
3. **Fix main.py** type errors
   - Environment variable handling
   - Import correct exception types from litellm
   - Fix string subscript operations
4. **Fix router_core.py** return types
   - Return empty lists instead of None
   - Fix chat_completions return type
   - Fix ModelResponse iteration

### Phase 3: Library Updates
5. **Update litellm imports** - check if exception types have changed
6. **Review ModelResponse** class - ensure it has proper `__aiter__` method

---

## 📊 Statistics

**Total LSP Errors:** ~40+
**Critical Issues:** 2 (router_config.yaml, detailed_logger.py)
**Type Safety Issues:** ~30+ (main.py, router_core.py)
**False Positives:** 1 (health_checker.py)

---

**These issues are pre-existing and were not introduced by type hint improvements.**
