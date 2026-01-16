# 🚀 Quick Start Guide: LLM API Proxy for AI Coding Tools

This guide will get your LLM API proxy running in 30 minutes or less, perfect for use with AI coding tools like Continue, Cursor, VS Code extensions, and more.

## 📋 What You Need

### 1. **API Keys from LLM Providers** (Choose 1-2 to start)
- **Gemini**: Get free API key at [Google AI Studio](https://makersuite.google.com/app/apikey)
- **OpenAI**: Get API key at [OpenAI Platform](https://platform.openai.com/api-keys)
- **Anthropic**: Get API key at [Anthropic Console](https://console.anthropic.com/)
- **Groq**: Get free API key at [Groq Console](https://console.groq.com/keys)

### 2. **GitHub Account** (for deployment)
- Sign up at [github.com](https://github.com) if you don't have one

### 3. **Render Account** (for free hosting)
- Sign up at [render.com](https://render.com) - free tier available

## 🎯 Step 1: Create Your Configuration File

I created a simplified `.env` file for you. Copy this and replace the placeholder values:

```env
# ==============================================================================
# ||                    YOUR PROXY CONFIGURATION                              ||
# ==============================================================================

# ------------------------------------------------------------------------------
# | [REQUIRED] Your Proxy Authentication Key                                  |
# ------------------------------------------------------------------------------
# Create a strong, unique secret key (you'll use this in your AI tools)
PROXY_API_KEY="my-awesome-secret-proxy-key-2024"

# ------------------------------------------------------------------------------
# | [CHOOSE YOUR PROVIDERS] Add Your API Keys                                 |
# ------------------------------------------------------------------------------

# --- Option A: Google Gemini (RECOMMENDED - Free tier available) ---
GEMINI_API_KEY_1="YOUR_GEMINI_API_KEY_HERE"

# --- Option B: OpenAI ---
# OPENAI_API_KEY_1="sk-your-openai-key-here"

# --- Option C: Anthropic Claude ---
# ANTHROPIC_API_KEY_1="sk-ant-your-anthropic-key-here"

# --- Option D: Groq (Very Fast, Free tier) ---
# GROQ_API_KEY_1="gsk_your-groq-key-here"

# ------------------------------------------------------------------------------
# | [OPTIONAL] G4F Fallback Providers (Free backup options)                   |
# ------------------------------------------------------------------------------
# These are free fallback providers when your main keys are exhausted
# Leave these blank for now - you can configure them later
G4F_API_KEY=""
G4F_MAIN_API_BASE=""
G4F_GROQ_API_BASE=""
G4F_GEMINI_API_BASE=""
```

**Save this as `.env` in your project folder.**

## 🧪 Step 2: Test Locally First (Optional but Recommended)

Before deploying, let's test it works on your computer:

### Option A: Use Pre-built Executable (Easiest)
1. Go to the [GitHub Releases page](https://github.com/ons96/LLM-API-Key-Proxy/releases)
2. Download the latest release for your operating system
3. Unzip and run `proxy_app.exe` (Windows) or `./proxy_app` (Mac/Linux)
4. It should start at `http://127.0.0.1:8000`

### Option B: Run from Source
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the proxy
python src/proxy_app/main.py
```

### Test Your Local Proxy
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-awesome-secret-proxy-key-2024" \
  -d '{"model": "gemini/gemini-2.5-flash", "messages": [{"role": "user", "content": "Hello! What is 2+2?"}]}'
```

You should get a response like: `{"choices":[{"message":{"content":"2 + 2 equals 4."}}]}`

## 🌐 Step 3: Deploy to Render (Free Hosting)

### 3.1: Fork the Repository
1. Go to [the GitHub repository](https://github.com/ons96/LLM-API-Key-Proxy)
2. Click **Fork** (top right) to create your own copy
3. Note your fork URL: `https://github.com/YOUR-USERNAME/LLM-API-Key-Proxy`

### 3.2: Create Web Service on Render
1. Go to [render.com](https://render.com) and log in
2. Click **New → Web Service**
3. Connect your GitHub account and select your forked repository
4. Fill in the form:
   - **Name**: `my-llm-proxy` (or any name you like)
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.proxy_app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`

### 3.3: Upload Your .env File
1. In your Render service dashboard, go to **Environment → Secret Files**
2. Click **Add Secret File**
3. **File Path**: `/` (leave as default)
4. **Contents**: Copy and paste your `.env` file contents
5. Click **Save**
6. Go to **Deploy → Manual Deploy → Deploy HEAD** to trigger a redeploy

### 3.4: Get Your Service URL
After deployment, note your service URL: `https://my-llm-proxy.onrender.com`

## 🔧 Step 4: Configure AI Coding Tools

### Continue (VS Code Extension)
Add to your `~/.continue/config.json`:
```json
{
  "models": [{
    "title": "My LLM Proxy - Gemini",
    "provider": "openai",
    "model": "gemini/gemini-2.5-flash",
    "apiBase": "https://my-llm-proxy.onrender.com/v1",
    "apiKey": "my-awesome-secret-proxy-key-2024"
  }]
}
```

### Cursor IDE
1. Go to **Settings → Features → Models**
2. Add new model:
   - **Provider**: OpenAI Compatible
   - **API Key**: `my-awesome-secret-proxy-key-2024`
   - **API Base URL**: `https://my-llm-proxy.onrender.com/v1`
   - **Model**: `gemini/gemini-2.5-flash`

### JanitorAI
1. Go to **API Settings**
2. Select **"Proxy"** mode
3. **API URL**: `https://my-llm-proxy.onrender.com/v1`
4. **API Key**: `my-awesome-secret-proxy-key-2024`
5. **Model**: `gemini/gemini-2.5-flash`

### Generic OpenAI-Compatible Tools
- **API Base URL**: `https://my-llm-proxy.onrender.com/v1`
- **API Key**: `my-awesome-secret-proxy-key-2024`
- **Model Format**: `provider/model_name` (e.g., `gemini/gemini-2.5-flash`)

## 🎮 Step 5: Available Models

Once your proxy is running, you can use these model formats:

### Gemini Models
- `gemini/gemini-2.5-flash` (fast, cost-effective)
- `gemini/gemini-2.5-pro` (higher quality)
- `gemini/gemini-1.5-flash-latest`

### OpenAI Models (if you add OpenAI key)
- `openai/gpt-4o`
- `openai/gpt-4o-mini`
- `openai/gpt-3.5-turbo`

### Anthropic Models (if you add Anthropic key)
- `anthropic/claude-3-5-sonnet`
- `anthropic/claude-3-haiku`

### Groq Models (if you add Groq key)
- `groq/llama-3.1-70b-versatile`
- `groq/mixtral-8x7b-32768`

## 🆘 Troubleshooting

### "401 Unauthorized" Error
- Check your `PROXY_API_KEY` matches exactly
- Ensure you're using `Authorization: Bearer YOUR_KEY` format

### "Model not found" Error
- Use the correct format: `provider/model_name` (e.g., `gemini/gemini-2.5-flash`)
- Check that you have the API key for that provider in your `.env`

### Service Sleeps (Free Tier)
- Render free tier sleeps after 15 minutes of inactivity
- First request after sleep takes ~30 seconds to wake up
- Upgrade to paid tier for always-on service

### Build Fails on Render
- Check the build logs in Render dashboard
- Ensure your `.env` file has the correct format
- Verify all required API keys are present

## 🎉 You're All Set!

Your LLM API proxy is now running and ready to use with any AI coding tool that supports OpenAI-compatible APIs. The proxy will automatically:
- Rotate between your API keys
- Handle rate limits gracefully
- Fall back to secondary providers if needed
- Log all requests for debugging

## 🔐 Security Notes
- Never share your `PROXY_API_KEY`
- Don't commit your `.env` file to version control
- Consider upgrading to Render's paid tier for better performance
- Monitor your API key usage on provider dashboards

## 📚 Need Help?
- Check the full [README.md](README.md) for advanced configuration
- See [Deployment guide.md](Deployment%20guide.md) for detailed deployment instructions
- Visit the [GitHub repository](https://github.com/ons96/LLM-API-Key-Proxy) for updates and issues

**Happy coding with your new unified LLM endpoint!** 🚀