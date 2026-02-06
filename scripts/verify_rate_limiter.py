import asyncio
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from proxy_app.rate_limiter import RateLimitTracker

async def test_rpm_limit():
    print('Testing RPM Limit...')
    tracker = RateLimitTracker()
    provider = 'test_provider'
    model = 'test_model'
    limits = {'rpm': 5}

    # Use up the limit
    for i in range(5):
        can_use = await tracker.can_use_provider(provider, model, limits)
        if not can_use:
            print(f'FAILED: Should be allowed (Request {i+1})')
            return False
        await tracker.record_request(provider, model)

    # Next request should fail
    can_use = await tracker.can_use_provider(provider, model, limits)
    if can_use:
        print('FAILED: Should be blocked by RPM limit')
        return False
    else:
        print('PASSED: Blocked by RPM limit')
        return True

async def test_cooldown():
    print('Testing Cooldown...')
    tracker = RateLimitTracker()
    provider = 'test_provider'
    model = 'test_model'
    
    # Trigger rate limit hit
    await tracker.record_rate_limit_hit(provider, model, retry_after=2.0)
    
    # Should be blocked
    can_use = await tracker.can_use_provider(provider, model, {})
    if can_use:
        print('FAILED: Should be blocked by cooldown')
        return False
        
    print('Waiting for cooldown...')
    await asyncio.sleep(2.1)
    
    # Should be allowed
    can_use = await tracker.can_use_provider(provider, model, {})
    if not can_use:
        print('FAILED: Should be allowed after cooldown')
        return False
        
    print('PASSED: Cooldown logic works')
    return True

async def main():
    success = True
    success &= await test_rpm_limit()
    success &= await test_cooldown()
    
    if success:
        print('\nAll Rate Limit Tests PASSED')
    else:
        print('\nSome Rate Limit Tests FAILED')
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
