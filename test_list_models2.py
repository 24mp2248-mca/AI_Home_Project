import os; from dotenv import load_dotenv; load_dotenv('backend/.env'); import google.generativeai as genai; genai.configure(api_key=os.getenv('GEMINI_API_KEY')); 
try: 
 for m in genai.list_models(): 
  if 'generateContent' in m.supported_generation_methods: print(m.name) 
except Exception as e: 
 print('ERROR:', str(e))
