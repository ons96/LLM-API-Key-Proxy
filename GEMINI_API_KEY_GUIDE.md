# 🔑 Getting Your Free Gemini API Key - Step by Step

This guide will help you get a free Gemini API key in 5 minutes. Gemini is perfect for getting started because it has a generous free tier and high rate limits.

## 📋 What You'll Get
- **15 requests per minute** (plenty for AI coding tools)
- **1,500 requests per day** (free tier)
- **High-quality models** including Gemini 2.5 Flash (very fast) and Gemini 2.5 Pro (high quality)
- **No credit card required**

## 🎯 Step-by-Step Instructions

### Step 1: Go to Google AI Studio
1. Open your web browser
2. Go to: **https://makersuite.google.com/app/apikey**
3. You may need to sign in with your Google account

### Step 2: Create API Key
1. Click **"Create API Key"** button
2. If prompted, select or create a Google Cloud Project:
   - Click **"Create API key in new project"** (recommended for beginners)
   - Or select an existing project if you have one
3. Your new API key will appear in a popup

### Step 3: Copy Your API Key
1. **Copy the API key immediately** (you won't see it again!)
2. It looks like: `AIzaSyC...` (about 40 characters long)
3. Paste it somewhere safe (like a text document)

### Step 4: Test Your Key (Optional but Recommended)
Let's make sure it works:

```bash
# Install curl if you don't have it (or use any HTTP client)
# Windows: Download from https://curl.se/windows/
# Mac: brew install curl
# Linux: sudo apt install curl

curl -X POST \
  https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": [{
      "parts": [{
        "text": "Hello! What is 2+2?"
      }]
    }]
  }' \
  -G -d "key=YOUR_API_KEY_HERE"
```

Replace `YOUR_API_KEY_HERE` with your actual API key.

**Expected response**: You should get a JSON response with "2 + 2 equals 4." or similar.

## 🎉 Success! What to Do Next

### If the test worked:
Great! Your API key is working. Now you can:

1. **Save your API key** somewhere secure
2. **Move to the next step**: Deploy your proxy to Render
3. **Use this key** in your `.env` file: `GEMINI_API_KEY_1="your-key-here"`

### If the test failed:
- Double-check you copied the key correctly
- Make sure you're not including extra spaces
- Try creating a new API key (you can have multiple)

## 🔧 What This Enables

With your Gemini API key, your proxy will be able to use these models:

### Fast & Cost-Effective
- `gemini/gemini-2.5-flash` - Great for most coding tasks
- `gemini/gemini-1.5-flash-latest` - Alternative fast model

### Higher Quality
- `gemini/gemini-2.5-pro` - Best for complex reasoning
- `gemini/gemini-1.5-pro-latest` - High-quality alternative

## 💡 Pro Tips

1. **Rate Limits**: You get 15 requests/minute, which is perfect for coding tools
2. **Daily Limits**: 1,500 requests/day should last most users a long time
3. **Multiple Keys**: You can create multiple Gemini API keys and add them all to your proxy for higher limits
4. **Free Forever**: Google's free tier has been very stable and reliable

## 🚨 Important Security Notes

- **Never share your API key** - it's like a password
- **Don't commit it to Git** - keep it in your `.env` file
- **Monitor usage** - check Google AI Studio for usage statistics
- **Rotate keys regularly** - create new ones periodically

## ❓ Troubleshooting

### "API key not valid"
- Double-check you copied the entire key
- Make sure there are no extra spaces at the beginning or end
- Try creating a new key

### "Quota exceeded"
- You've hit the daily limit (1,500 requests)
- Wait until the next day, or create a new API key
- Consider upgrading to paid tier for higher limits

### "Permission denied"
- Make sure you created the API key in a Google Cloud Project
- Try creating a new API key

## ✅ Next Steps

Once you have your Gemini API key:
1. **Keep it safe** - save it to a secure location
2. **Continue to Render deployment** - use my Quick Start Guide
3. **Configure your AI tools** - I'll show you how next

**Ready to move on to deployment? Let me know and I'll walk you through setting up Render!** 🚀