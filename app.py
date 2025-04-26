from flask import Flask, render_template, request, jsonify
from safety_assistant import SafetyAssistant
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Initialize safety assistant chatbot
try:
    assistant = SafetyAssistant()
    logger.info("Safety Assistant initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Safety Assistant: {str(e)}")
    assistant = None

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    if assistant is None:
        return jsonify({'response': 'Sorry, the safety assistant is currently unavailable. Please try again later.'}), 500
    
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'response': 'Please provide a message.'}), 400
            
        response = assistant.process_message(user_message)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        return jsonify({'response': 'Sorry, I encountered an error. Please try again.'}), 500

@app.route('/get_next_chunk', methods=['POST'])
def get_next_chunk():
    try:
        remaining_chunks = request.json.get('remaining_chunks', [])
        if not remaining_chunks:
            return jsonify({'chunk': '', 'has_more': False, 'remaining_chunks': []})
            
        next_chunk = remaining_chunks[0]
        remaining = remaining_chunks[1:]
        
        return jsonify({
            'chunk': next_chunk,
            'has_more': len(remaining) > 0,
            'remaining_chunks': remaining
        })
    except Exception as e:
        logger.error(f"Error getting next chunk: {str(e)}")
        return jsonify({'chunk': '', 'has_more': False, 'remaining_chunks': []}), 500

if __name__ == '__main__':
    app.run(debug=True) 