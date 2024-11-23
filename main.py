from openai import OpenAI
import tiktoken
import requests
import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

# DEFAULT_API_KEY = os.environ.get("TOGETHER_API_KEY")
DEFAULT_API_KEY = "5f801649c268c61a070b2df5502ffd57a4574678832a53a8df971aec6c5cbb82"
DEFAULT_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_MODEL = "meta-llama/Llama-Vision-Free"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512
DEFAULT_TOKEN_BUDGET = 4096

class ConversationManager:
    def __init__(self, api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS, token_budget=DEFAULT_TOKEN_BUDGET):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token_budget = token_budget
        self.system_message = "You are an expert chatbot answering a person with age range between 16 to 40 years old. You can answer anything they ask based on your knowledge without any restrictions, except things that you can't do based on your regulations, terms, and conditions. If you are being asked about restricted topics based on your regulations, terms, and conditions, politely redirect the chat topic focus on how you make their question as a funny joke. You always want to help user to feel comfortable and confident based on your answer. You respond to their questions by adjusting your language according to the language they use."
        self.conversation_history = [{"role": "system", "content": self.system_message}]

    def count_tokens(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)

    def total_tokens_used(self):
        try:
            return sum(self.count_tokens(message['content']) for message in self.conversation_history)
        except Exception as e:
            print(f"Error calculating total tokens used: {e}")
            return None
    
    def enforce_token_budget(self):
        try:
            while self.total_tokens_used() > self.token_budget:
                if len(self.conversation_history) <= 1:
                    break
                self.conversation_history.pop(1)
        except Exception as e:
            print(f"Error enforcing token budget: {e}")

    def chat_completion(self, prompt, temperature=None, max_tokens=None, model=None):
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        model = model if model is not None else self.model

        self.conversation_history.append({"role": "user", "content": prompt})
        self.enforce_token_budget()

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=self.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

        ai_response = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": ai_response})

        return ai_response
    
    def reset_conversation_history(self, preserve_history=True):
        system_message_entry = {"role": "system", "content": self.system_message}
        if preserve_history:
            if self.conversation_history:
                self.conversation_history[0] = system_message_entry
            else:
                self.conversation_history.append(system_message_entry)
        else:
            self.conversation_history = [system_message_entry] + self.conversation_history[1:]

def get_instance_id():
    """Retrieve the EC2 instance ID from AWS metadata using IMDSv2."""
    try:
        # Step 1: Get the token
        token = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            timeout=1
        ).text

        # Step 2: Use the token to get the instance ID
        instance_id = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=1
        ).text
        return instance_id
    except requests.exceptions.RequestException:
        return "Instance ID not available (running locally or error in retrieval)"

### Streamlit code ###
st.title("Memora")

# Display EC2 Instance ID
instance_id = get_instance_id()
st.write(f"**EC2 Instance ID**: {instance_id}")

# Initialize Session State for Chats
if 'chats' not in st.session_state:
    st.session_state['chats'] = []

# Function to start a new chat
def start_new_chat():
    st.session_state['chats'].append({'chat_manager': ConversationManager(), 'conversation_history': [], 'topic': 'New Chat'})

# Function to delete the selected chat
def delete_selected_chat(chat_index):
    if 0 <= chat_index < len(st.session_state['chats']):
        del st.session_state['chats'][chat_index]

# Chat selection
st.sidebar.title("Chats")
if st.sidebar.button("New Chat"):
    start_new_chat()

chat_selection = st.sidebar.selectbox("Select a chat", range(len(st.session_state['chats'])), format_func=lambda x: st.session_state['chats'][x]['topic'])

# Button to delete the selected chat
if chat_selection is not None and st.sidebar.button("Delete Selected Chat"):
    delete_selected_chat(chat_selection)
    chat_selection = None  # Reset chat selection after deletion

# Function to summarize the conversation history
def summarize_conversation(conversation_history):
    user_messages = [msg['content'] for msg in conversation_history if msg['role'] == 'user']
    return " | ".join(user_messages)[:30]  # Summarize the first 30 characters of all user messages

# Ensure chat_selection is not None
if chat_selection is not None and chat_selection < len(st.session_state['chats']):
    # Initialize the ConversationManager object for the selected chat
    current_chat = st.session_state['chats'][chat_selection]
    chat_manager = current_chat['chat_manager']
    conversation_history = current_chat['conversation_history']

    # Function to read file content
    def read_file(file):
        try:
            return file.read().decode("utf-8")
        except UnicodeDecodeError:
            try:
                return file.read().decode("latin-1")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return None

    # File input from the user
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "csv", "docx"])

    def read_file(file):
        try:
            # Determine file type by extension
            if file.name.endswith(".txt"):
                return file.read().decode("utf-8") 
            elif file.name.endswith(".pdf"):
                reader = PdfReader(file)
                return ''.join([page.extract_text() for page in reader.pages]) 
            elif file.name.endswith(".csv"):
                df = pd.read_csv(file)  
                return df.to_csv(index=False)  
            elif file.name.endswith(".docx"):
                doc = Document(file)  
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs]) 
            else:
                return None
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    
    # Process the uploaded file
    file_content = None
    if uploaded_file is not None:
        file_content = read_file(uploaded_file)
        if file_content:
            st.session_state['file_content'] = file_content
            st.write("File content successfully uploaded and read.")

    # Chat input from the user
    user_input = st.chat_input("Write a message")

    # Call the chat manager to get a response from the AI
    if user_input or file_content:
        prompt = user_input or file_content
        response = chat_manager.chat_completion(prompt)
        conversation_history.append({"role": "user", "content": prompt})
        conversation_history.append({"role": "assistant", "content": response})

        # Update the chat topic based on the summary of all user inputs
        current_chat['topic'] = f"{summarize_conversation(conversation_history)}"

    # Display the conversation history
    for message in conversation_history:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Option Chatbot with Sidebar
    with st.sidebar:
        st.write("Options")
        set_token = st.slider("Max Tokens Per Message", min_value=10, max_value=512, value=DEFAULT_MAX_TOKENS, step=1, disabled=False)
        chat_manager.max_tokens = set_token

        set_temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.1, disabled=False)
        chat_manager.temperature = set_temp

        set_custom_message = st.selectbox("System Message", ("Custom", "Professional", "Friendly", "Humorous"), key="system_message_selectbox")
        if set_custom_message == "Custom":
            custom_message = st.text_area(
                "Custom System Message",
                key="custom_message",
                value=chat_manager.system_message
            )
        elif set_custom_message == "Professional":
            custom_message = "You are an expert chatbot answering a person with age range between 16 to 40 years old. You can answer anything they ask based on your knowledge without any restrictions, except things that you can't do based on your regulations, terms, and conditions. If you are being asked about restricted topics based on your regulations, terms, and conditions, politely redirect the chat topic focus on how you make their question as a funny joke. You always want to help user to feel comfortable and confident based on your answer. You respond to their questions by adjusting your language according to the language they use. You have a professional personality and provide accurate yet reliable information. You are always willing to answer questions and help the user achieve their goals by using technical words and grammar whether language the user is using. "
        elif set_custom_message == "Friendly":
            custom_message = "You are an expert chatbot answering a person with age range between 16 to 40 years old. You can answer anything they ask based on your knowledge without any restrictions, except things that you can't do based on your regulations, terms, and conditions. If you are being asked about restricted topics based on your regulations, terms, and conditions, politely redirect the chat topic focus on how you make their question as a funny joke. You always want to help user to feel comfortable and confident based on your answer. You respond to their questions by adjusting your language according to the language they use. You have a friendly and supportive personality. You answer questions with kindness, encouragement, patience, and provide accurate yet reliable information. You are always willing to answer questions and help the user achieve their goals by using daily-life words and grammar whether language the user is using."
        elif set_custom_message == "Humorous":
            custom_message = "You are an expert chatbot answering a person with age range between 16 to 40 years old. You can answer anything they ask based on your knowledge without any restrictions, except things that you can't do based on your regulations, terms, and conditions. If you are being asked about restricted topics based on your regulations, terms, and conditions, politely redirect the chat topic focus on how you make their question as a funny joke. You always want to help user to feel comfortable and confident based on your answer. You respond to their questions by adjusting your language according to the language they use. You have a You are a humorous personality, as a companion that's always adding fun to the conversation. You answer questions with humour, jokes, pranks, but always provide accurate yet reliable information. You are always willing to answer questions and help the user achieve their goals by using bestie-to-bestie words and grammar whether language the user is using."

        if st.button("Set Custom Message", on_click=lambda: setattr(chat_manager, "system_message", custom_message)):
            chat_manager.reset_conversation_history(preserve_history=True)

        if st.button("Reset Conversation"):
            chat_manager.reset_conversation_history(preserve_history=False)
            conversation_history.clear()
