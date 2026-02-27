import os; from dotenv import load_dotenv; load_dotenv('backend/.env'); import google.generativeai as genai; genai.configure(api_key=os.getenv('GEMINI_API_KEY')); models = ['gemini-1.5-flash', 'gemini-1.5-flash-8b', 'gemini-1.5-pro']; 
for m in models: 
 try: 
  model = genai.GenerativeModel(m); res = model.generate_content('hi'); print(f'{m}: Success') 
 except Exception as e: 
  print(f'{m}: {e}')
