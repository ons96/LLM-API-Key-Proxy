#!/usr/bin/env python3
"""
Simple test script to verify your Groq API key works.
Run this after getting your API key to make sure it's working.

Usage:
    python test_groq_key.py YOUR_GROQ_API_KEY_HERE
"""

import sys
import requests
import json

def test_groq_api_key(api_key):
    """Test if a Groq API key is working."""
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "user", 
                "content": "Hello! Please respond with just the word 'success' if you can see this message."
            }
        ],
        "max_tokens": 50,
        "temperature": 0.1
    }
    
    try:
        print("🔄 Testing your Groq API key...")
        print(f"📡 Making request to: {url}")
        print(f"🤖 Using model: llama-3.1-8b-instant")
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract the response text
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content'].strip()
                print("✅ SUCCESS! Your Groq API key is working!")
                print(f"🤖 Groq response: {content}")
                
                # Show some additional info
                if 'usage' in result:
                    usage = result['usage']
                    print(f"📊 Tokens used: {usage.get('total_tokens', 'N/A')}")
                
                return True
            else:
                print("❌ Unexpected response format")
                print(f"Response: {json.dumps(result, indent=2)}")
                return False
                
        elif response.status_code == 401:
            print("❌ Unauthorized - API key may be invalid")
            print(f"Response: {response.text}")
            return False
            
        elif response.status_code == 429:
            print("⚠️  Rate limit exceeded - you've hit the token limit")
            print("Wait a minute and try again, or consider upgrading your plan")
            print(f"Response: {response.text}")
            return False
            
        elif response.status_code == 400:
            print("❌ Bad Request - Check your request format")
            print(f"Response: {response.text}")
            return False
            
        else:
            print(f"❌ Error {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"Response text: {response.text}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("❌ Usage: python test_groq_key.py YOUR_GROQ_API_KEY_HERE")
        print("\nExample:")
        print("python test_groq_key.py gsk_1234567890abcdef...")
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    # Basic validation
    if len(api_key) < 20:
        print("❌ API key seems too short. Make sure you copied the entire key.")
        sys.exit(1)
    
    if not api_key.startswith("gsk_"):
        print("⚠️  Warning: Groq API keys usually start with 'gsk_'. Make sure you have the right key.")
        response = input("Continue anyway? (y/n): ").lower().strip()
        if response != 'y':
            sys.exit(1)
    
    success = test_groq_api_key(api_key)
    
    if success:
        print("\n🎉 Your Groq API key is working perfectly!")
        print("📝 Next steps:")
        print("   1. Save your API key securely")
        print("   2. Follow the QUICK_START_GUIDE.md to deploy your proxy")
        print("   3. Add this key to your .env file as GROQ_API_KEY_1")
        print("   4. Use models like 'groq/llama-3.1-8b-instant' or 'groq/llama-3.1-70b-versatile'")
        
        print("\n🚀 Why Groq is great for AI coding:")
        print("   • Super fast responses (usually under 1 second)")
        print("   • High quality models (Llama 3.1)")
        print("   • Generous free tier (30,000 tokens/minute)")
        print("   • Perfect for coding tools like Continue, Cursor")
        
    else:
        print("\n❌ There was a problem with your API key.")
        print("📝 Common solutions:")
        print("   1. Make sure you copied the entire key")
        print("   2. Check that you didn't include extra spaces")
        print("   3. Try creating a new API key in Groq Console")
        print("   4. Verify the key is for Groq (starts with 'gsk_')")

if __name__ == "__main__":
    main()