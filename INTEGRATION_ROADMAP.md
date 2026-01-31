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
