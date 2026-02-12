import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load env from .env file explicitly
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

class HomePlannerChatbot:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = None
        self.chat_session = None
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            # Use 'gemini-flash-latest' which is a valid alias in the user's account
            self.model = genai.GenerativeModel('gemini-flash-latest')
        else:
            print("WARNING: GEMINI_API_KEY not found. Chatbot will return fallback responses.")

    def process_message(self, message, context):
        """
        Gemini-powered chat logic with STRICT context separation.
        """
        if not self.model:
            return {
                "text": "I'm currently offline (Missing API Key). Please set GEMINI_API_KEY in backend/.env to wake me up!",
                "action": None
            }

        msg = message.strip()
        
        # --- Context Preparation ---
        page = context.get('page', 'unknown')
        cost = context.get('estimated_cost', 0)
        area = context.get('total_area', 0)
        
        # --- Dynamic System Prompt based on Page ---
        if page == 'visualization':
            system_prompt = f"""
            You are an AI Design Assistant for a 3D Home Visualization tool.
            
            CONTEXT:
            - User is on the **3D VISUALIZATION PAGE**.
            - House Area: {area} sq.m
            
            STRICT RULES:
            1. YOU MUST NOT ANSWER QUESTIONS ABOUT COST, PRICE, OR BUDGET. politely refuse and tell the user to go to the Cost Estimation page.
            2. FOCUS ONLY ON: Visuals, Design, Furniture, Colors, and 3D environment.
            3. ALLOWED ACTIONS: You can modify the scene using the 'action' field.
            
            AVAILABLE ACTIONS:
            - change_color (value: hex_code/name, target: walls/roof)
            - apply_theme (value: modern/rustic/cyberpunk/luxury)
            - add_furniture (value: sofa/table/bed/chair)
            - add_feature (value: garden/pool)
            
            INSTRUCTIONS:
            - If user asks to change design (e.g. "blue walls", "add sofa"), generate the ACTION.
            - If user asks for a style (e.g. "make it rustic", "cyberpunk theme"), use 'apply_theme'.
            - If user asks about cost ("how much is this?", "budget?"), REPLY: "I specialize in design here. Please visit the Cost Estimation page for pricing details."
            - Be concise and helpful.
            
            OUTPUT FORMAT: Single JSON object.
            {{
                "text": "Response text...",
                "action": {{ "type": "...", "value": "..." }} OR null
            }}
            """
            
        elif page == 'cost_estimation':
            system_prompt = f"""
            You are an AI Construction Cost Consultant.
            
            CONTEXT:
            - User is on the **COST ESTIMATION PAGE**.
            - Estimated Total Cost: ${cost}
            - House Area: {area} sq.m
            
            STRICT RULES:
            1. YOU MUST NOT PERFORM VISUAL ACTIONS (No changing colors, no adding furniture).
            2. YOU MUST NOT ANSWER QUESTIONS ABOUT 3D MANIPULATION.
            3. FOCUS ONLY ON: Budget, Material Costs, Construction Timeline, and Financial breakdown.
            
            INSTRUCTIONS:
            - Explain the costs based on the provided total estimates.
            - Breakdown logic: Kitchen ~20%, Living ~15%, Structure ~35%.
            - If user asks to "change wall color" or "add furniture", REPLY: "I cannot modify the 3D model from this page. Please return to the defined Visualization page to make design changes."
            - If user asks about currency, explain 1 USD ~= 84 INR.
            
            OUTPUT FORMAT: Single JSON object.
            {{
                "text": "Response text...",
                "action": null
            }}
            """
            
        else:
            # Fallback for unknown pages
            system_prompt = f"""
            You are a helpful Home Planning Assistant.
            Current Page: {page}.
            Please guide the user to either the Visualization page (for design) or Cost Estimation page (for budgeting).
            OUTPUT FORMAT: Single JSON object {{ "text": "...", "action": null }}
            """

        try:
            # Generate valid JSON response
            response = self.model.generate_content(
                f"{system_prompt}\n\nUSER MESSAGE: {message}\n\nRESPONSE (JSON):",
                generation_config={"response_mime_type": "application/json"}
            )
            
            # Parse JSON
            data = json.loads(response.text)
            return {
                "text": data.get("text", "I'm thinking..."),
                "action": data.get("action")
            }

        except Exception as e:
            print(f"Gemini Error: {e}")
            return {
                "text": f"I'm having trouble connecting to my AI brain right now. Error details: {str(e)}",
                "action": None
            }

chatbot = HomePlannerChatbot()

def get_response(message, context):
    return chatbot.process_message(message, context)
