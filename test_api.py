import os; from dotenv import load_dotenv; load_dotenv('backend/.env'); import google.generativeai as genai; genai.configure(api_key=os.getenv('GEMINI_API_KEY')); model = genai.GenerativeModel('gemini-1.5-flash'); 
try: 
 res = model.generate_content('hello'); print('SUCCESS') 
except Exception as e: 
 print('ERROR:', str(e))
