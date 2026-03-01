import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load env from .env file explicitly
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path, override=True)

class HomePlannerChatbot:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        self.model_name = 'llama-3.1-8b-instant'
        
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            print("WARNING: GROQ_API_KEY not found. Chatbot will return fallback responses.")

    def process_message(self, message, context):
        """
        Groq-powered chat logic with STRICT context separation.
        """
        if not self.client:
            return {
                "text": "I'm currently offline (Missing API Key). Please set GROQ_API_KEY in backend/.env to wake me up!",
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
            
            OUTPUT FORMAT: You MUST return a valid JSON object.
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
            
            OUTPUT FORMAT: You MUST return a valid JSON object.
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
            OUTPUT FORMAT: You MUST return a valid JSON object {{ "text": "...", "action": null }}
            """

        try:
            # Generate valid JSON response using Groq
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"USER MESSAGE: {message}\n\nRESPONSE (JSON):"
                    }
                ],
                model=self.model_name,
                response_format={"type": "json_object"},
            )
            
            resp_text = chat_completion.choices[0].message.content
            
            # Parse JSON
            data = json.loads(resp_text)
            return {
                "text": data.get("text", "I'm thinking..."),
                "action": data.get("action")
            }

        except Exception as e:
            print(f"Groq Error: {e}")
            is_rate_limit = "429" in str(e)
            if is_rate_limit:
                base_mock = "I am currently overloaded by the free tier rate limit! Please wait 60 seconds and try your request again."
            else:
                base_mock = "I am currently operating in offline/fallback mode due to an API error. Please check the GROQ_API_KEY."

            if page == 'cost_estimation':
                mock_text = f"Fallback Mode: The estimated cost for your {area} sq.m house is roughly ${cost}. ({base_mock})"
            elif page == 'visualization':
                mock_text = f"Fallback Mode: I cannot change the model right now. {base_mock}"
            else:
                mock_text = base_mock
                
            return {
                "text": mock_text,
                "action": None
            }

chatbot = HomePlannerChatbot() #create a single instance of the chatbot to maintain session state if needed

def get_response(message, context):
    return chatbot.process_message(message, context)
