
## Push Confirmation - Tue Jan 27 15:26:16 EST 2026
- Successfully pushed 5 atomic commits to origin/main.
- Final HEAD: 86a50c4
- Included scoring logic, model rankings, provider updates, VPS tools, and verification tests.
- Verified .env was not included (ignored by .gitignore).
- Commits:
  - 86a50c4 test: add verification scripts for concurrency and quick proxy check
  - a4067e1 feat(deploy): add VPS deployment tools and documentation
  - 738d971 feat(api): include virtual models in OpenAI-compatible model list
  - 078bb33 feat(providers): update Cerebras models and improve Google OAuth callback
  - e84ec04 refactor(router): implement composite scoring and update 2026 model rankings
- Streaming Fallback Pattern: To support fallback during streaming, the router must implement a generator method (e.g., `_stream_with_fallback`) that iterates through candidates and yields chunks, rather than returning the provider's generator directly. This allows catching exceptions raised during the stream consumption and seamlessly switching to the next candidate.
