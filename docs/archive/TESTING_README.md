# LLM Gateway Testing - Quick Start

## What You Have

1. **test_vps_gateway.sh** - Main testing script (runs on VPS gateway)
2. **run_tests.bat** - Windows launcher (opens WSL and runs tests)

## How to Use

### Option 1: Windows (Easiest - Double-click)

1. **Edit** `test_vps_gateway.sh` and update the VPS_HOST at line 11:
   ```bash
   VPS_HOST="10.1.0.1"  # Change to your VPS IP
   ```

2. **Double-click** `run_tests.bat`
   - This will open WSL terminal
   - Run all tests automatically
   - Show you results

### Option 2: WSL Terminal

1. Open WSL terminal
2. Run:
   ```bash
   cd ~/CodingProjects/LLM-API-Key-Proxy
   ./test_vps_gateway.sh
   ```

### Option 3: SSH Directly

If you prefer to SSH manually:
```bash
ssh owens@<your-vps-ip>
```

Then navigate to gateway and test:
```bash
cd ~/path/to/LLM-API-Key-Proxy
# Run the same tests as in test_vps_gateway.sh
```

## What Tests Run

The script automatically tests:

✅ **TEST 1**: Server connectivity
✅ **TEST 2**: Model list (2218+ models)
✅ **TEST 3**: Virtual models with fallback
   - coding-elite
   - coding-fast
   - chat-smart
   - chat-fast
   - chat-rp
✅ **TEST 4**: Direct provider access
   - Groq Llama 3.3 70B
   - Groq Llama 3.1 8B Instant
   - Gemini Pro
   - G4F GPT-4
✅ **TEST 5**: Fallback chain behavior
✅ **TEST 6**: Rate limiting (rapid requests)

## Expected Results

**All tests should PASS with green ✓ marks:**
- ✓ Server is ONLINE
- ✓ Virtual models working (returns "Hello to you" etc.)
- ✓ Direct providers working
- ✓ Fallback tries multiple providers (logs show "trying next")
- ✓ Rate limits detected and router falls back

## Troubleshooting

### If Tests Fail:

1. **"Server is OFFLINE"**
   - Check VPS is running
   - Check VPS_HOST in script is correct
   - Try: `ping <vps-ip>`

2. **"Failed to fetch models"**
   - Gateway might not be started on VPS
   - Check: VPS has gateway running on port 9099

3. **"✗ FAILED" on models**
   - Check VPS logs: `tail -f ~/llm_proxy.log`
   - API keys might be missing
   - Provider might be down

## Next Steps After Testing

**If all tests pass:**
1. Gateway is ready for OpenCode
2. Configure OpenCode:
   - Base URL: `http://<vps-ip>:9099`
   - Model: `coding-elite` or `coding-fast`
   - API Key: `test` (or any value)

**If tests fail:**
1. Check VPS gateway logs
2. Restart gateway on VPS if needed
3. Verify .env file has API keys
