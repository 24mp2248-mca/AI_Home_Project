import os
from backend.chatbot import chatbot

try:
    chatbot.process_message('add a garden', {'page': 'visualization'})
except Exception as e:
    with open('error_msg.txt', 'w', encoding='utf-8') as f:
        f.write(str(e))
