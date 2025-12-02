#!/usr/bin/env python3
"""
Test Gemini API with the api key from api_keys.json
"""

import json
from pathlib import Path
# import google.generativeai as genai

# Load API key
api_keys_path = Path(__file__).parent.parent / "api_keys.json"
with open(api_keys_path, "r") as f:
    api_keys = json.load(f)

# api_key = api_keys["gemini_api_key_eva"]
# genai.configure(api_key=api_key)

# model = genai.GenerativeModel('gemini-2.5-flash')
# response = model.generate_content('explain how ai works in a few words')

# print(response.text)

from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=api_keys["gemini_api_key_eva"])

response = client.models.generate_content(
    model="gemini-3-pro-preview", contents="Explain how AI works in a few words"
)
print(response.text)