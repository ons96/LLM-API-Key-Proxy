#!/usr/bin/env python3
"""
Simple test script to verify your Gemini API key works.
Run this after getting your API key to make sure it's working.

Usage:
    python test_gemini_key.py YOUR_API_KEY_HERE
"""

import sys
import requests
import json

def test_gemini_api_key(api_key):
    """Test if a Gemini API key is working."""
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{
            "parts": [{
                "text": "Hello! Please respond with just the word 'success' if you can see this message."
            }]
        }]
    }
    
    # Add API key as query parameter
    params = {
        "key": api_key
    }
    
    try:
        print("🔄 Testing your Gemini API key...")
        print(f"📡 Making request to: {url}")
        
        response = requests.post(url, headers=headers, params=params, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract the response text
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0]['content']['parts'][0]['text']
                print("✅ SUCCESS! Your API key is working!")
                print(f"🤖 Gemini response: {content}")
                print(f"📊 Full response: {json.dumps(result, indent=2)}")
                return True
            else:
                print("❌ Unexpected response format")
                print(f"Response: {result}")
                return False
                
        elif response.status_code == 400:
            print("❌ Bad Request - Check your API key format")
            print(f"Response: {response.text}")
            return False
            
        elif response.status_code == 403:
            print("❌ Forbidden - API key may be invalid or quota exceeded")
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
        print("❌ Usage: python test_gemini_key.py YOUR_API_KEY_HERE")
        print("\nExample:")
        print("python test_gemini_key.py AIzaSyC1234567890abcdef...")
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    # Basic validation
    if len(api_key) < 20:
        print("❌ API key seems too short. Make sure you copied the entire key.")
        sys.exit(1)
    
    if not api_key.startswith("AIza"):
        print("⚠️  Warning: Gemini API keys usually start with 'AIza'. Make sure you have the right key.")
    
    success = test_gemini_api_key(api_key)
    
    if success:
        print("\n🎉 Your Gemini API key is working perfectly!")
        print("📝 Next steps:")
        print("   1. Save your API key securely")
        print("   2. Follow the QUICK_START_GUIDE.md to deploy your proxy")
        print("   3. Add this key to your .env file as GEMINI_API_KEY_1")
    else:
        print("\n❌ There was a problem with your API key.")
        print("📝 Common solutions:")
        print("   1. Make sure you copied the entire key")
        print("   2. Check that you didn't include extra spaces")
        print("   3. Try creating a new API key in Google AI Studio")
        print("   4. Verify the key is for Gemini (not other Google services)")

if __name__ == "__main__":
    main()