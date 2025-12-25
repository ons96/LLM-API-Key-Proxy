# 🚀 Getting Your Free Groq API Key - Super Easy!

Groq is perfect for AI coding tools - it's fast, free, and very easy to set up. You'll have your API key in 2 minutes!

## 🎯 Why Groq is Great
- ✅ **Completely free** with generous limits
- ✅ **Super fast** inference (great for coding tools)
- ✅ **Easy signup** (no complex verification)
- ✅ **High quality models** (Llama, Mixtral, Gemma)
- ✅ **Perfect for AI coding** tools like Continue, Cursor

## 📊 What You Get
- **30,000 tokens/minute** (plenty for coding)
- **Free tier** that lasts a very long time
- **Fast responses** (usually under 1 second)
- **No credit card required**

## 🎯 Step-by-Step Instructions

### Step 1: Go to Groq Console
1. **Open your browser**
2. **Go to**: https://console.groq.com/keys
3. **Click "Sign Up"** (top right)

### Step 2: Create Your Account
1. **Enter your email** and create a password
   - OR click **"Continue with Google"** for faster signup
2. **Verify your email** if required
3. **Complete the signup process**

### Step 3: Create Your API Key
1. **Look for "API Keys"** section (usually on the main page)
2. **Click "Create API Key"** or **"+" button**
3. **Give it a name** like "My Coding Key" or "Proxy Key"
4. **Click "Create"**

### Step 4: Copy Your API Key
1. **Copy the key immediately** (you won't see it again!)
2. **It starts with**: `gsk_` (about 50 characters long)
3. **Save it somewhere safe**

## 🧪 Test Your Key (Optional)

Let's make sure it works:

### Option A: Use the Test Script
```bash
python test_groq_key.py YOUR_GROQ_API_KEY_HERE
```

### Option B: Manual Test with curl
```bash
curl -X POST "https://api.groq.com/openai/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_GROQ_API_KEY_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-70b-versatile",
    "messages": [{"role": "user", "content": "Hello! What is 2+2?"}],
    "max_tokens": 50
  }'
```

**Expected response**: JSON with "2 + 2 equals 4." or similar.

## 🎮 Available Models

With Groq, you can use these excellent models:

### Fast & Efficient
- `groq/llama-3.1-8b-instant` - Very fast, great for most tasks
- `groq/mixtral-8x7b-32768` - Balanced speed/quality

### High Quality
- `groq/llama-3.1-70b-versatile` - High quality, still fast
- `groq/llama-3.1-405b-reasoning` - Best quality (if available)

### Code-Focused
- `groq/deepseek-r1-distill-llama-70b` - Excellent for reasoning

## 💡 Pro Tips

1. **Rate Limits**: 30,000 tokens/minute is very generous
2. **Speed**: Usually responds in under 1 second
3. **Quality**: Llama 3.1 models are excellent for coding
4. **Multiple Keys**: You can create multiple keys if needed

## 🔐 Security Notes

- **Keep your key secret** - never share it publicly
- **Don't commit to Git** - keep it in your `.env` file
- **Monitor usage** - check Groq console for statistics
- **Rotate keys** - create new ones periodically

## ❓ Troubleshooting

### "Invalid API key"
- Double-check you copied the entire key
- Make sure it starts with `gsk_`
- Try creating a new key

### "Rate limit exceeded"
- You've hit the token limit for now
- Wait a minute and try again
- Consider upgrading for higher limits

### "Model not found"
- Use the exact model names listed above
- Some models might not be available in free tier

## 🎉 Success!

Once you have your Groq API key:
1. **Save it securely**
2. **Add it to your .env file**: `GROQ_API_KEY_1="your-key-here"`
3. **Continue with deployment** using the Quick Start Guide
4. **Use models like**: `groq/llama-3.1-70b-versatile`

## 📝 Quick Reference

**API Endpoint**: `https://api.groq.com/openai/v1/`
**Key Format**: `gsk_...` (starts with gsk_)
**Best Models**: 
- Fast: `groq/llama-3.1-8b-instant`
- Quality: `groq/llama-3.1-70b-versatile`

**Ready to move on to deployment? Groq makes everything so much simpler!** 🚀