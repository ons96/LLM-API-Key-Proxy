# 🚀 Render Deployment Guide - Step by Step

This guide will deploy your LLM API proxy to Render's free hosting in about 15 minutes.

## 📋 What You'll Need
- ✅ Your `.env` file (you already created this!)
- ✅ GitHub account (for the repository)
- ✅ Render account (free at render.com)

## 🎯 Step 1: Fork the Repository

### 1.1: Go to GitHub
1. **Open your browser**
2. **Go to**: https://github.com/ons96/LLM-API-Key-Proxy
3. **Sign in** to your GitHub account (create one if needed)

### 1.2: Fork the Repository
1. **Click the "Fork" button** (top right, near the star button)
2. **Choose your account** as the destination
3. **Click "Create fork"**
4. **Wait for it to complete** (usually 10-30 seconds)

### 1.3: Note Your Fork URL
Your fork will be at: `https://github.com/YOUR-USERNAME/LLM-API-Key-Proxy`
**Save this URL** - you'll need it for Render!

## 🌐 Step 2: Create Render Account

### 2.1: Sign Up for Render
1. **Go to**: https://render.com
2. **Click "Sign Up"**
3. **Choose "Sign up with GitHub"** (recommended)
4. **Authorize Render** to access your GitHub account

## 🔧 Step 3: Create Web Service

### 3.1: Create New Service
1. **In Render Dashboard**, click **"New +"**
2. **Select "Web Service"**
3. **Click "Build and deploy from a Git repository"**
4. **Click "Next"**

### 3.2: Connect Repository
1. **Find your forked repository** in the list (LLM-API-Key-Proxy)
2. **Click "Connect"** next to it

### 3.3: Configure Web Service
Fill in these exact settings:

**Basic Settings:**
- **Name**: `my-llm-proxy` (or any name you like)
- **Region**: Choose closest to you (e.g., Oregon for US West)
- **Branch**: `main`
- **Runtime**: `Python 3`

**Build and Deploy:**
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn src.proxy_app.main:app --host 0.0.0.0 --port $PORT`

**Instance Type:**
- **Select**: `Free`

### 3.4: Create Service
1. **Click "Create Web Service"**
2. **Wait for initial build** (2-5 minutes)

## 🔐 Step 4: Upload .env File

### 4.1: Navigate to Environment
1. **In your service dashboard**, click **"Environment"**
2. **Click "Secret Files"** tab
3. **Click "Add Secret File"**

### 4.2: Configure Secret File
- **File Path**: `/` (leave as default)
- **Contents**: Copy and paste the ENTIRE contents of your `.env` file

**Important**: Make sure you paste the full content, including both lines:
```env
PROXY_API_KEY="your-key"
GROQ_API_KEY_1="your-groq-key"
```

### 4.3: Save and Redeploy
1. **Click "Save"**
2. **Go to "Deploy"** tab
3. **Click "Manual Deploy"**
4. **Select "Deploy HEAD"**
5. **Wait for deployment** (2-3 minutes)

## ✅ Step 5: Test Your Deployment

### 5.1: Get Your Service URL
After deployment completes:
1. **Note your service URL**: `https://my-llm-proxy.onrender.com`
2. **The URL will be** in your service dashboard

### 5.2: Test with curl
```bash
curl -X POST https://your-service-name.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{"model": "groq/llama-3.1-8b-instant", "messages": [{"role": "user", "content": "Hello! What is 2+2?"}]}'
```

**Replace:**
- `your-service-name` with your actual service name
- `your-proxy-api-key` with your PROXY_API_KEY from .env

### 5.3: Expected Response
You should get a JSON response like:
```json
{
  "choices": [{
    "message": {
      "content": "2 + 2 equals 4."
    }
  }]
}
```

## 🎮 Step 6: Configure AI Coding Tools

### Continue (VS Code)
Add to your `~/.continue/config.json`:
```json
{
  "models": [{
    "title": "My LLM Proxy - Groq",
    "provider": "openai",
    "model": "groq/llama-3.1-8b-instant",
    "apiBase": "https://your-service-name.onrender.com/v1",
    "apiKey": "your-proxy-api-key"
  }]
}
```

### Cursor IDE
1. **Settings** → **Features** → **Models**
2. **Add new model**:
   - **Provider**: OpenAI Compatible
   - **API Key**: Your PROXY_API_KEY
   - **API Base URL**: `https://your-service-name.onrender.com/v1`
   - **Model**: `groq/llama-3.1-8b-instant`

### Generic OpenAI-Compatible Tools
- **API Base URL**: `https://your-service-name.onrender.com/v1`
- **API Key**: Your PROXY_API_KEY
- **Model**: `groq/llama-3.1-8b-instant`

## 🔍 Troubleshooting

### Build Fails
- **Check build logs** in Render dashboard
- **Verify requirements.txt** exists in repository
- **Ensure repository** is your fork, not the original

### 401 Unauthorized
- **Check your PROXY_API_KEY** matches exactly
- **Verify you're using** `Authorization: Bearer YOUR_KEY`

### Service Sleeps (Free Tier)
- **First request** after sleep takes ~30 seconds
- **Upgrade to paid tier** for always-on service

### Model Not Found
- **Use correct format**: `groq/llama-3.1-8b-instant`
- **Check your Groq key** is valid and has quota

## 🎉 Success!

Once working, your proxy will:
- ✅ **Automatically rotate** between your API keys
- ✅ **Handle rate limits** gracefully
- ✅ **Provide fallback** if providers fail
- ✅ **Work with any OpenAI-compatible tool**

## 📊 Available Models
- **Fast**: `groq/llama-3.1-8b-instant`
- **Quality**: `groq/llama-3.1-70b-versatile`
- **Balanced**: `groq/mixtral-8x7b-32768`

## 🔐 Security Notes
- ✅ Your `.env` file is **securely stored** as a secret
- ✅ **Never commit** .env to version control
- ✅ **Monitor usage** in Groq console
- ✅ **Keep proxy key secret** - don't share it

## 🚀 Next Steps
1. **Test with your favorite AI coding tool**
2. **Add more providers** (OpenAI, Anthropic, etc.) to your .env
3. **Consider upgrading** to Render's paid tier for better performance

**Your LLM API proxy is now live and ready to use!** 🎯