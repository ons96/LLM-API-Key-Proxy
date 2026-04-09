I'll analyze this routing architecture issue and provide a fix. This is a common pattern where legacy or direct routing bypasses a centralized router abstraction.

## Root Cause Analysis

The issue typically occurs when:
1. **Mixed routing paradigms**: Some routes use the centralized `RouterCore` (for middleware, logging, auth), while others use direct framework decorators (e.g., `@app.route`, `@router.get`)
2. **Inconsistent middleware application**: Bypassing RouterCore means missing cross-cutting concerns (authentication, logging, error handling)
3. **Code duplication**: Route registration logic scattered between `RouterCore` and direct registrations

## The Fix

### Before (Buggy main.py)
```python
from fastapi import FastAPI, HTTPException
from router_core import RouterCore, RouteHandler

app = FastAPI()
router_core = RouterCore(app)

# BUG: Direct registration bypasses RouterCore's middleware pipeline
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# BUG: Direct registration missing auth middleware
@app.post("/api/data")
async def create_data(payload: dict):
    # Missing authentication check that RouterCore would enforce
    return {"id": 123, "data": payload}

# CORRECT: Uses RouterCore (has logging, auth, error handling)
router_core.register(RouteHandler(
    path="/api/users",
    method="GET",
    handler=get_users,
    middleware=[auth_required, audit_log]
))

@app.get("/metrics")
async def metrics():
    # Also bypasses RouterCore
    return {"requests": 1000}
```

### After (Fixed main.py)
```python
from fastapi import FastAPI, HTTPException
from router_core import RouterCore, RouteHandler
from middleware import auth_required, audit_log, health_check_middleware
from handlers import get_users, create_data_handler, health_handler, metrics_handler

app = FastAPI()
router_core = RouterCore(app)

def initialize_routes():
    """Centralized route registration through RouterCore."""
    
    # FIX: Moved health check to RouterCore with proper middleware
    router_core.register(RouteHandler(
        path="/health",
        method="GET",
        handler=health_handler,
        middleware=[health_check_middleware],
        tags=["monitoring"]
    ))

    # FIX: Consolidated data endpoint with auth middleware
    router_core.register(RouteHandler(
        path="/api/data",
        method="POST",
        handler=create_data_handler,
        middleware=[auth_required, audit_log],
        tags=["data"]
    ))

    # FIX: Users endpoint remains properly registered
    router_core.register(RouteHandler(
        path="/api/users",
        method="GET",
        handler=get_users,
        middleware=[auth_required, audit_log],
        tags=["users"]
    ))

    # FIX: Metrics endpoint now uses RouterCore for consistency
    router_core.register(RouteHandler(
        path="/metrics",
        method="GET",
        handler=metrics_handler,
        middleware=[audit_log],  # Now has logging
        tags=["monitoring"]
    ))

# Initialize all routes through the core
initialize_routes()

# Remove any direct app registrations to prevent bypass
# All routing now flows through: Request -> RouterCore -> Middleware -> Handler
```

## Alternative Fix (If RouterCore wraps FastAPI/Flask)

If `RouterCore` is meant to be a wrapper/proxy:

```python
# main.py
from router_core import RouterCore
from handlers import user_routes, system_routes

def create_app():
    # RouterCore now owns the FastAPI/Flask instance
    router = RouterCore()
    
    # All registration happens through RouterCore
    router.register_routes(user_routes)
    router.register_routes(system_routes)
    
    # FIX: Remove direct access to underlying app to prevent bypass
    # return router.app  # Old way - exposes internal app
    return router.get_wrapped_app()  # New way - controlled access

app = create_app()
```

## What Changed and Why

### 1. **Eliminated Direct Decorator Registration**
- **Removed**: `@app.get("/health")` and similar direct registrations
- **Why**: These bypassed `RouterCore`'s middleware pipeline, causing security and observability gaps

### 2. **Centralized Route Registration**
- **Added**: `initialize_routes()` function that consolidates all route definitions
- **Why**: Single source of truth for routing logic; easier to audit and maintain

### 3. **Enforced Middleware Consistency**
- **Changed**: All routes now explicitly declare middleware through `RouterCore`
- **Why**: Prevents "shadow routes" that work but lack authentication, logging, or rate limiting

### 4. **Removed Dormant Code**
- **Removed**: Any hardcoded route handlers in `main.py` 
- **Why**: Business logic should live in `handlers/` modules; `main.py` should only bootstrap

## Verification Checklist

After applying the fix, verify:
- [ ] No `@app.route`, `@router.get`, or similar decorators remain in `main.py`
- [ ] All HTTP endpoints appear in `RouterCore`'s registration logs
- [ ] Authentication middleware fires for all protected routes (including previously bypassed ones)
- [ ] Access logs are generated consistently for all endpoints

This fix ensures architectural consistency and prevents security holes caused by middleware bypass.
