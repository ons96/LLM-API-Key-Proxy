00609| - Migration effort vs resource savings trade-off
00610| - Only proceed if benchmarking shows significant RAM reduction (50%+)
00611| 
00612| ---
00613| 
00614| ## New Requirements - Dynamic Scoring & Provider Management (Jan 2026)
00615| 
00616| ### Revised Scoring Formula (Quality-First)
00617| 
00618| **Problem**: Current formula (60% agentic + 30% TPS + 10% availability) allows fast but weak models like Llama 3.3 70B (65.2 SWE-bench) to outrank slow but strong models.
00619| 
00620| **Solution**: Change weights to prioritize coding quality:
00621| - **Agentic Coding Score**: 80% (primary factor)
00622| - **TPS Score**: 15% (secondary - only matters within quality tier)
00623| - **Availability Score**: 5% (tiebreaker)
00624| 
00625| **Minimum Score Threshold**:
00626| - coding-elite: Minimum 70.0 SWE-bench score
00627| - coding-smart: Minimum 65.0 SWE-bench score
00628| - Models below threshold: Last resort only (absolute final fallback)
00629| 
00630| **Free Model Baseline**:
00631| - OpenCode Zen free models as minimum quality bar:
00632|   - minimax m2.1 free
00633|   - trinity large preview
00634|   - kimi k2.5 free
00635|   - glm-4.7 free
00636|   - big pickle
00637| - Only include models better than worst available free model
00638| - Exception: If no free models available, use whatever works
00639| 
00640| ### Score Handling & Fallbacks
00641| 
00642| **Missing/NA Score Handling**:
00643| - Try to match model by API name normalization
00644| - If no score found: Use median score of similar model family
00645| - If still unknown: Place at end of fallback chain with warning
00646| - Never exclude a model entirely due to missing score (user might want it)
00647| 
00648| **Reasoning Effort Variations**:
00649| - Different reasoning levels (thinking vs non-thinking) have different performance
00650| - Store scores per model + reasoning_level combination
00651| - Example: claude-opus-4.5 vs claude-opus-4.5-thinking-32k = different scores
00652| 
00653| ### New Provider: OpenCode Zen
00654| 
00655| **API Endpoint**: `https://opencode.ai/zen/v1/chat/completions`
00656| 
00657| **Free Tier Benefits**:
00658| - Unlimited daily usage
00659| - Rate limits under heavy load only
00660| - No API key required (but adding support if user provides one)
00661| 
00662| **Available Models** (free tier):
00663| - minimax m2.1 free
00664| - trinity large preview
00665| - kimi k2.5 free
00666| - glm-4.7 free
00667| - big pickle
00668| 
00669| **Implementation Tasks**:
00670| - Add OpenCode Zen provider adapter
00671| - Configure rate limits (dynamic based on load)
00672| - Add to coding-elite/coding-smart as baseline
00673| 
00674| ### Event-Driven Dynamic Scoring
00675| 
00676| **Problem**: Fixed 5-minute recalculation wastes resources when nothing changed.
00677| 
00678| **Solution**: Event-driven recalculation:
00679| - Trigger on: API call completion (update TPS), rate limit detected, health status change, provider model list change
00680| - Manual trigger via `/v1/admin/recalculate-rankings` endpoint
00681| - Maximum frequency: Once per minute (debounce rapid events)
00682| 
00683| **Resource Usage**:
00684| - Memory: ~20MB for ranked list cache
00685| - CPU: <50ms per recalculation
00686| - Runs only when needed
00687| 
00688| ### Chat Model Rankings (chat-smart, chat-fast)
00689| 
00690| **Data Source**: Artificial Analysis API/Scraper
00691| - URL: https://artificialanalysis.ai/leaderboards/providers
00692| - API endpoint: TBD (check if they have public API)
00693| 
00694| **chat-smart Virtual Model**:
00695| - Primary: Intelligence/reasoning score (Chatbot Arena, MMLU, reasoning benchmarks)
00696| - Order: Best to worst by intelligence
00697| - Assumption: User willing to wait for best response
00698| 
00699| **chat-fast Virtual Model**:
00700| - Primary: Efficiency ratio = Intelligence ÷ Response Time
00701| - Order: Best efficiency (smart + fast) to worst
00702| - Example: Might prefer GPT-5.2 medium over GPT-5.2 xHigh if nearly as smart but 3x faster
00703| 
00704| **Reasoning Effort Handling**:
00705| - Store reasoning_level (none, low, medium, high, xhigh) in model config
00706| - Different scores for different reasoning levels
00707| - Allow user to specify preferred reasoning level per virtual model
00708| 
00709| ### Provider Model List Refresh
00710| 
00711| **Problem**: Provider offerings change over time (add/remove models, pricing changes).
00712| 
00713| **Solution**: Periodic refresh:
00714| - Frequency: Every 6 hours (configurable)
00715| - Fetch latest model list from each provider
00716| - Detect: New models, removed models, pricing changes, free tier changes
00717| - Update scoring data automatically
00718| - Log changes for review
00719| 
00720| **Implementation**:
00721| - Background async task
00722| - Cache model lists to avoid API spam
00723| - Update virtual_models.yaml dynamically (or runtime cache)
00724| 
00725| ### Testing Strategy
00726| 
00727| **Recommended Order**:
00728| 1. **Test current gateway first** - Verify existing functionality works
00729| 2. **Add OpenCode Zen provider** - Expand provider pool
00730| 3. **Implement revised scoring** - 80/15/5 weights with minimum thresholds
00731| 4. **Add event-driven recalculation** - Optimize resource usage
00732| 5. **Implement chat model rankings** - Add chat-smart/chat-fast support
00733| 6. **Add provider refresh** - Keep model lists current
00734| 
00735| ### Future Roadmap Items
00736| 
00737| #### AI Chatbot Integration
00738| - Gateway as API provider for AI chatbots
00739| - Virtual models optimized for chatbot use cases
00740| - Reasoning effort configuration per conversation
00741| - Session-based model selection
00742| - Cost tracking per chatbot deployment
00743| 
00744| ---
00745| 
00746| ## Quick Reference: Implementation Order
00747| 
00748| 1. Test current gateway functionality
00749| 2. Add OpenCode Zen provider support
00750| 3. Update scoring formula (80/15/5 weights)
00751| 4. Implement minimum score thresholds
00752| 5. Add event-driven recalculation
00753| 6. Add score handling fallbacks (missing/NA)
00754| 7. Implement chat model rankings (Artificial Analysis)
00755| 8. Add provider model list refresh
00756| 9. Create unit tests for new scoring
00757| 10. Update documentation

## New Requirements - Rate Limiting & Advanced Features (Jan 2026)

### Rate Limit Tracking Per Model

**Problem**: Free providers have strict rate limits. Using the same model multiple times in quick succession can cause rejections.

**Solution**: Track recent usage per model/provider:
- Track last_used_timestamp per model/provider
- Track requests_per_minute count (rolling window)
- Check rate limits before selecting fallback model
- Skip models that would exceed rate limits

**Implementation**:
```python
class RateLimitTracker:
    - track_last_used(provider, model) -> timestamp
    - track_requests_in_window(provider, model, window_minutes=1) -> count
    - check_rate_limit_ok(provider, model) -> bool
    - skip_if_recently_used(models_list, min_interval_seconds=30)
```

**Database Schema Addition**:
```sql
CREATE TABLE model_usage_tracking (
    provider TEXT,
    model TEXT,
    last_used TIMESTAMP,
    requests_last_minute INTEGER,
    requests_last_5_minutes INTEGER,
    PRIMARY KEY (provider, model)
);
```

### Rate/Usage Limits in Scoring Formula

**Enhanced Scoring with Rate Limits**:
```
Score = (AgenticScore × 0.70) + (TPS × 0.15) + (Availability × 0.05) + (RateLimitScore × 0.10)

Where RateLimitScore = 
- Unlimited/High limits: 100
- Medium limits (100-1000 RPM): 70
- Low limits (10-100 RPM): 40
- Very low limits (<10 RPM): 10
```

**Priority Boost for High-Limit Models**:
- OpenCode Zen free models: +20% score boost (unlimited usage)
- Groq free tier: +15% score boost (high limits)
- Cerebras: +15% score boost (high limits)
- Low-limit providers: score penalty

### Aggregate Chat Scores from Multiple Sources

**Problem**: Current chat rankings only use Artificial Analysis. More sources = better accuracy.

**Solution**: Aggregate scores from:
1. Artificial Analysis (primary)
2. Arena.ai (Chatbot Arena) - overall + creative writing leaderboards
3. LiveBench - reasoning + language scores
4. MMLU scores

**Normalization Formula**:
```
AggregateIntelligence = 
    (ArtificialAnalysis × 0.40) +
    (Arena.ai × 0.30) +
    (LiveBench × 0.20) +
    (MMLU × 0.10)
```

**Implementation Tasks**:
- [ ] Scraper for Arena.ai leaderboards
- [ ] Scraper for LiveBench scores
- [ ] Score normalization across different scales
- [ ] Weighted aggregation formula
- [ ] Update chat_model_rankings.yaml with aggregated scores

### Individual Model Fallback Support

**Problem**: User selects specific model (e.g., "claude-sonnet-4.5"), but one provider fails. No automatic fallback to other providers with same model.

**Solution**: Create dynamic virtual models for every available model:
- User requests: "claude-sonnet-4.5"
- Gateway creates (or uses cached) virtual model for claude-sonnet-4.5
- Fallback chain: All providers offering claude-sonnet-4.5, ranked by score
- Auto-fallback if one provider fails

**Implementation Approach**:
```python
class IndividualModelFallback:
    def create_dynamic_virtual_model(self, model_id: str) -> VirtualModel:
        # Find all providers offering this model
        providers = self.find_providers_with_model(model_id)
        # Rank by scoring formula
        ranked = self.rank_providers(providers, model_id)
        # Return virtual model with fallback chain
        return VirtualModel(model_id, ranked)
```

**Complexity**: HIGH
- Requires mapping model IDs across providers (different naming)
- Requires caching virtual models to avoid recreation overhead
- Requires updating when providers add/remove models

**Status**: Future enhancement (not immediate priority)

### Threshold Review: Is 65.0 Too High?

**Current Thresholds**:
- coding-elite: 70.0
- coding-smart: 65.0

**Models Above 65.0** (from model_rankings.yaml):
- Anthropic Claude Opus 4.5: 74.4 ✓
- OpenAI o3-pro: 74.1 ✓
- Anthropic Claude Sonnet 4.5: 70.6 ✓
- Google Gemini 2.5 Pro: 73.2 ✓
- OpenAI GPT-4.5: 71.2 ✓
- DeepSeek V3.5: 68.9 ✓
- Alibaba Qwen 3 Max: 66.5 ✓
- Moonshot AI Kimi K2: 65.8 ✓
- Mistral Codestral 2: 65.3 ✓
- XAI Grok 3.5: 65.1 ✓
- OpenCode Zen MiniMax: ~65.0 (estimated)

**Models Below 65.0** (excluded from coding-smart):
- Llama 3.3 70B: 65.2 (borderline)
- Cohere Command R7B: 60.8 ✗
- Mistral Medium 3: 58.3 ✗
- Qwen 2.5 72B: 56.9 ✗
- GPT-4o: 56.7 ✗
- Grok 2: 54.5 ✗
- Gemini 1.5 Pro: 53.8 ✗
- Llama 3.1 405B: 51.1 ✗

**Conclusion**: 65.0 threshold excludes lower-performing models but keeps good ones. Could be lowered to 60.0 to include more variety if needed.

