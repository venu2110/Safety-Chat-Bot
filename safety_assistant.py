import os
import google.generativeai as genai
from datetime import datetime
import json
import re
from dotenv import load_dotenv
import logging
import random

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Successfully configured Gemini API")
except Exception as e:
    logger.error(f"Error configuring Gemini API: {str(e)}")
    raise

class SafetyAssistant:
    def __init__(self):
        try:
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            logger.info("Successfully initialized Gemini model")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            raise
        
        self.conversation_history = []
        self.last_question_asked = None
        self.question_count = 0
        self.safety_categories = [
            "Emergency Response",
            "Fire Safety",
            "Personal Security",
            "Health & Hygiene",
            "Road Safety",
            "Home Safety",
            "Workplace Safety",
            "Environmental Safety",
            "Cyber Security",
            "First Aid"
        ]
        
        # Varied questions to avoid repetition
        self.varied_questions = [
            "Would you like to learn more about emergency procedures?",
            "Should we discuss personal safety measures?",
            "Would you like to know about first aid basics?",
            "Do you need information about fire safety?",
            "Would you like to learn about cyber security?",
            "Should we focus on home safety tips?",
            "Would you like to know about road safety?",
            "Do you want to learn about workplace safety?",
            "Would you like to know about environmental safety?",
            "Should we discuss health and hygiene practices?"
        ]
        
        # Track previously suggested ideas to avoid repetition
        self.previous_suggestions = set()
        self.safety_themes = [
            "Emergency Preparedness",
            "Prevention First",
            "Stay Alert",
            "Safety First",
            "Be Prepared",
            "Protect Yourself",
            "Secure Environment",
            "Health Conscious",
            "Safe Living",
            "Risk Management"
        ]
        
        # Safety tips and guidelines
        self.safety_tips = {
            "Emergency Response": [
                "Keep emergency numbers handy",
                "Know your evacuation routes",
                "Have a first aid kit ready",
                "Create an emergency contact list"
            ],
            "Fire Safety": [
                "Install smoke detectors",
                "Keep fire extinguishers accessible",
                "Plan escape routes",
                "Never leave cooking unattended"
            ],
            "Personal Security": [
                "Stay aware of surroundings",
                "Travel in groups when possible",
                "Keep valuables secure",
                "Share your location with trusted contacts"
            ],
            "Health & Hygiene": [
                "Wash hands frequently",
                "Maintain social distance",
                "Wear masks in crowded places",
                "Stay hydrated and eat well"
            ]
        }
    
    def generate_content_safely(self, prompt):
        """Safely generate content with error handling"""
        try:
            logger.debug(f"Sending prompt to Gemini API: {prompt[:100]}...")
            response = self.model.generate_content(prompt)
            logger.debug("Successfully received response from Gemini API")
            return response.text
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}. Please try again or rephrase your question."

    def get_varied_question(self):
        """Get a varied question that hasn't been asked recently"""
        if self.question_count >= 2:
            available_questions = [q for q in self.varied_questions if q != self.last_question_asked]
            if available_questions:
                question = random.choice(available_questions)
            else:
                question = self.generate_new_question()
            self.question_count = 0
        else:
            question = random.choice(self.varied_questions)
            
        if question == self.last_question_asked:
            self.question_count += 1
        else:
            self.question_count = 0
            
        self.last_question_asked = question
        return question

    def generate_new_question(self):
        """Generate a new, unique question to avoid repetition"""
        themes = random.sample(self.safety_themes, 1)[0]
        
        new_questions = [
            f"Would you like to learn about {themes}?",
            f"Should we discuss {random.choice(self.safety_categories)}?",
            f"Would you like to know about {random.choice(['emergency procedures', 'prevention tips', 'safety measures'])}?",
            f"Should we focus on {random.choice(['personal', 'home', 'workplace'])} safety?",
            f"Would you like to learn about {random.choice(['first aid', 'emergency response', 'prevention'])}?",
            f"Should we discuss {random.choice(['cyber', 'road', 'environmental'])} safety?",
            f"Would you like to know about {random.choice(['health', 'hygiene', 'wellness'])} practices?",
            f"Should we talk about {random.choice(['emergency kits', 'safety equipment', 'prevention tools'])}?"
        ]
        
        return random.choice(new_questions)

    def chunk_response(self, response):
        """Split a response into smaller, manageable chunks"""
        chunks = []
        current_chunk = []
        current_length = 0
        max_chunk_length = 150  # Maximum characters per chunk
        
        # Split response into lines
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # If adding this line would exceed max length, start a new chunk
            if current_length + len(line) > max_chunk_length and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(line)
            current_length += len(line)
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

    def process_message(self, user_message):
        """Process user messages and generate appropriate responses"""
        try:
            # Add message to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            
            # Check for specific safety-related queries
            if any(keyword in user_message.lower() for keyword in ["emergency", "help", "danger", "unsafe"]):
                response = self.handle_emergency_query(user_message)
            elif any(keyword in user_message.lower() for keyword in ["safety tips", "guidelines", "advice"]):
                response = self.provide_safety_tips(user_message)
            elif any(keyword in user_message.lower() for keyword in ["first aid", "medical", "health"]):
                response = self.handle_health_query(user_message)
            else:
                prompt = f"""
                You are a helpful safety assistant. Help the user with their safety-related queries.
                Available safety categories: {', '.join(self.safety_categories)}
                
                User query: {user_message}
                
                Provide a clear and structured response that:
                1. Starts with a brief, engaging introduction (1 sentence)
                2. Uses emojis to make the response more visually appealing
                3. Organizes information in clear subpoints with bullet points
                4. Includes specific safety tips and guidelines
                5. Uses clear and direct language
                6. Emphasizes important safety information
                7. Does not include a question at the end
                
                Format your response as:
                - Main response with emoji (1 sentence)
                - Subpoint 1: [emoji] Detail
                - Subpoint 2: [emoji] Detail
                - Subpoint 3: [emoji] Detail
                - No paragraphs or long text blocks
                """
                response = self.generate_content_safely(prompt)
                
                # Add a varied question at the end
                response += "\n\n" + self.get_varied_question()
            
            # Split response into chunks
            response_chunks = self.chunk_response(response)
            
            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Return the first chunk with a flag indicating more chunks are available
            return {
                "chunk": response_chunks[0],
                "has_more": len(response_chunks) > 1,
                "remaining_chunks": response_chunks[1:] if len(response_chunks) > 1 else []
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "chunk": f"I apologize, but I encountered an error: {str(e)}. Please try again or rephrase your question.",
                "has_more": False,
                "remaining_chunks": []
            }

    def handle_emergency_query(self, user_message):
        """Handle emergency-related queries"""
        prompt = f"""
        The user has indicated an emergency situation: {user_message}
        
        Provide immediate safety guidance that:
        1. Starts with clear emergency instructions
        2. Lists immediate actions to take
        3. Includes emergency contact numbers
        4. Provides step-by-step guidance
        5. Emphasizes calm and clear thinking
        
        Format as:
        🚨 EMERGENCY RESPONSE
        - Immediate Action 1
        - Immediate Action 2
        - Emergency Contacts
        - Next Steps
        """
        return self.generate_content_safely(prompt)

    def provide_safety_tips(self, user_message):
        """Provide relevant safety tips based on the query"""
        # Determine the most relevant category
        category = None
        for cat in self.safety_categories:
            if cat.lower() in user_message.lower():
                category = cat
                break
        
        if not category:
            category = random.choice(self.safety_categories)
        
        tips = self.safety_tips.get(category, [
            "Stay alert and aware of your surroundings",
            "Keep emergency contacts readily available",
            "Follow safety protocols and guidelines",
            "Report any safety concerns immediately"
        ])
        
        response = f"🔒 {category} Tips:\n\n"
        for tip in tips:
            response += f"• {tip}\n"
        
        response += "\n" + self.get_varied_question()
        return response

    def handle_health_query(self, user_message):
        """Handle health and medical-related queries"""
        prompt = f"""
        The user has a health-related query: {user_message}
        
        Provide health and safety guidance that:
        1. Starts with immediate health advice
        2. Lists relevant health precautions
        3. Includes basic first aid steps if applicable
        4. Emphasizes when to seek professional medical help
        5. Provides clear health safety guidelines
        
        Format as:
        🏥 HEALTH & SAFETY GUIDANCE
        - Immediate Advice
        - Health Precautions
        - First Aid Steps (if applicable)
        - When to Seek Medical Help
        """
        return self.generate_content_safely(prompt) 