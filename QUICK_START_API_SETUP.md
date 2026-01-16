# 🚀 Quick Start: Set Up Your LLM API Proxy for AI Coding Tools

I'll walk you through everything needed to get this running as an API endpoint for your AI coding tools. This is designed for complete beginners!

## 📋 What You'll Need

### 1. API Keys (You'll Need to Get These Yourself)
You'll need at least one API key from an LLM provider. Here are the easiest options:

**🆓 Free Options:**
- **Groq**: Sign up at [console.groq.com](https://console.groq.com) - Very fast, generous free tier
- **OpenRouter**: Sign up at [openrouter.ai](https://openrouter.ai) - Access to many models
- **Gemini**: Get free key at [Google AI Studio](https://makersuite.google.com/app/apikey)

**💰 Paid Options (Better Performance):**
- **OpenAI**: Sign up at [platform.openai.com](https://platform.openai.com)
- **Anthropic**: Sign up at [console.anthropic.com](https://console.anthropic.com)

### 2. A Render Account (Free)
- Go to [render.com](https://render.com) and sign up (free)
- You'll connect your GitHub account

### 3. Your Own GitHub Account
- If you don't have one, create at [github.com](https://github.com) (free)

## 🎯 Step 1: Test Locally First (Recommended)

Before deploying, let's test it on your computer to make sure everything works:

### Download and Setup
1. **Download the project**: 
   - Go to [GitHub Releases](https://github.com/ons96/LLM-API-Key-Proxy/releases)
   - Download the latest release for your operating system
   - Extract the ZIP file

2. **Create your .env file**:
   - Copy the `.env.example` file and rename it to `.env`
   - Open `.env` in a text editor (Notepad, VS Code, etc.)
   - Replace the placeholder values with your actual API keys

### Sample .env File (Replace with Your Real Keys)
```env
# Your proxy authentication key (create any strong password)
PROXY_API_KEY="my-awesome-proxy-key-12345"

# Provider API keys (get these from the provider websites)
GEMINI_API_KEY_1="your-gemini-key-here"
GROQ_API_KEY_1="your-groq-key-here"
OPENAI_API_KEY_1="your-openai-key-here"
```

### Test the Proxy
1. **Run the proxy**: Double-click `proxy_app.exe` (Windows) or run `./proxy_app` (Mac/Linux)
2. **Test it**: Open your browser and go to `http://127.0.0.1:8000`
   - You should see "API Key Proxy is running"
3. **Test with a simple request**:
   ```bash
   curl -X POST http://127.0.0.1:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer my-awesome-proxy-key-12345" \
     -d '{"model": "groq/llama3-8b-8192", "messages": [{"role": "user", "content": "Say hello!"}]}'
   ```

**If this works locally, you're ready to deploy!**

## 🌐 Step 2: Deploy to Render (Free Hosting)

Render will host your API endpoint online so AI tools can access it from anywhere.

### 2.1 Fork the Repository
1. Go to [GitHub](https://github.com/ons96/LLM-API-Key-Proxy)
2. Click **Fork** (top right) - this creates your own copy
3. Note your fork URL: `https://github.com/YOUR-USERNAME/LLM-API-Key-Proxy`

### 2.2 Deploy to Render
1. **Log into Render**:
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"

2. **Connect your repository**:
   - Choose "Build and deploy from a Git repository"
   - Connect your GitHub account
   - Select your forked repository

3. **Configure the deployment**:
   - **Name**: `my-llm-proxy` (or any name you like)
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.proxy_app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free` (for now)

4. **Add your environment variables**:
   - After creating the service, go to "Environment" tab
   - Add these variables:
     ```
     PROXY_API_KEY=my-awesome-proxy-key-12345
     GEMINI_API_KEY_1=your-gemini-key-here
     GROQ_API_KEY_1=your-groq-key-here
     ```

5. **Deploy**:
   - Click "Create Web Service"
   - Wait 5-10 minutes for deployment to complete
   - Note your service URL (e.g., `https://my-llm-proxy.onrender.com`)

### 2.3 Test Your Deployed API
```bash
curl -X POST https://your-service-name.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-awesome-proxy-key-12345" \
  -d '{"model": "groq/llama3-8b-8192", "messages": [{"role": "user", "content": "Say hello!"}]}'
```

## 🤖 Step 3: Configure AI Coding Tools

Now you can use this endpoint in AI coding tools. Here are examples for popular tools:

### Continue.dev (VS Code Extension)
Add to your `~/.continue/config.json`:
```json
{
  "models": [
    {
      "title": "My LLM Proxy - Groq",
      "provider": "openai",
      "model": "groq/llama3-8b-8192",
      "apiBase": "https://your-service-name.onrender.com/v1",
      "apiKey": "my-awesome-proxy-key-12345"
    },
    {
      "title": "My LLM Proxy - Gemini",
      "provider": "openai", 
      "model": "gemini/gemini-1.5-flash",
      "apiBase": "https://your-service-name.onrender.com/v1",
      "apiKey": "my-awesome-proxy-key-12345"
    }
  ]
}
```

### Cursor IDE
In Cursor settings:
- **API Provider**: Custom
- **Base URL**: `https://your-service-name.onrender.com/v1`
- **API Key**: `my-awesome-proxy-key-12345`
- **Model**: `groq/llama3-8b-8192` (or any supported model)

### JanitorAI / SillyTavern
- **API URL**: `https://your-service-name.onrender.com/v1`
- **API Key**: `my-awesome-proxy-key-12345`
- **Model**: `groq/llama3-8b-8192`

## 🔧 Available Models

Once deployed, you can see all available models by visiting:
`https://your-service-name.onrender.com/v1/models`

**Popular models to try:**
- `groq/llama3-8b-8192` (Fast, good for coding)
- `groq/llama3-70b-8192` (More powerful)
- `gemini/gemini-1.5-flash` (Google's fast model)
- `gemini/gemini-1.5-pro` (Google's powerful model)
- `openai/gpt-4o` (OpenAI's latest)

## 💰 Cost Estimate

**Free Tier Limitations:**
- Render free tier sleeps after 15 minutes of inactivity
- First request after sleep takes ~30 seconds to wake up
- Good for personal use and testing

**Monthly Costs:**
- **Render**: Free (or $7/month for always-on)
- **Groq**: Free tier (very generous)
- **Gemini**: Free tier (15 requests/minute)
- **OpenRouter**: $5-20/month depending on usage

## 🚨 Important Notes

1. **Keep your API keys secret**: Never share your `.env` file or API keys
2. **The free tier sleeps**: Your endpoint will "wake up" after the first request
3. **Provider rate limits**: Each provider has different limits
4. **Model availability**: Some models may not be available from all providers

## 🆘 Troubleshooting

**"401 Unauthorized"**: Check that your PROXY_API_KEY matches exactly
**"Model not found"**: Make sure you're using the format `provider/model-name`
**"Connection refused"**: Your Render service might be sleeping - wait 30 seconds and try again
**"Build failed"**: Check Render logs for dependency issues

## 🎉 You're Done!

Once deployed, you have a single API endpoint that works with multiple AI providers, automatic failover, and can be used by any AI coding tool that supports OpenAI-compatible APIs!

**Your API endpoint**: `https://your-service-name.onrender.com/v1`