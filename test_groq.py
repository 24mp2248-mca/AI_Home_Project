import os
import json
from backend.chatbot import chatbot

try:
    print(chatbot.process_message('add a garden', {'page': 'visualization'}))
except Exception as e:
    print(e.message if hasattr(e, 'message') else str(e))
