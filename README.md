# Memora Chatbot <a href="http://team2-alb-127132480.ap-southeast-1.elb.amazonaws.com/" target="_blank"> <img align="center" src="https://raw.githubusercontent.com/MFRDS/Team2-CDK-Chatbot/refs/heads/main/assets/ai_icon.png" height="30" width="30" /></a>

Welcome to the Memora Chatbot repository! This project is made by Team 2 CendekiAwan RevoU to demonstrate how we build an intelligent chatbot using AWS Architecture, featuring Streamlit and SQL Alchemy. The chatbot provides real-time responses and allows users to manage and retrieve past conversations. User can also choose how Memora would behave in three mode: Professional, Friendly, or Humorous. Uploading a file is available and users can ask anything about it to Memora.

<img src=https://raw.githubusercontent.com/MFRDS/Team2-CDK-Chatbot/refs/heads/main/assets/memora.png width="70%" height="70%"/>

## Features

- **Interactive UI**: Built with Streamlit for a user-friendly interface.
- **Conversation Management**: Switch between bot behavior and recall past conversations using SQL Alchemy.
- **Advanced Language Model**: Utilizes Llama and OpenAI for conversational AI tasks.
- **Read File Capability**: Provides file reading capability and simple explanation.

## Installation

To get started with the project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/TafazzulAffan/Memora-Chatbot.git
    ```

2. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit app**:
    ```bash
    streamlit run main.py
    ```

2. **Interact with the chatbot**:
    - Open the app in your browser.
    - Use the sidebar to start a new conversation or load a past conversation.
    - Type your messages in the input field and receive real-time responses from the chatbot.
    - Upload files by drag, drop, or browse file with certain file type.

## Appendix

- **AWS EC2 Instance Architecture**: Cloud-based infrastructure for scalable and secure application deployment.
- **Streamlit**: Framework for creating interactive web apps directly from Python scripts.
- **SQL Alchemy**: Powerful library for database interactions using ORM and raw SQL.
- **OpenAI**: AI models for natural language processing and conversational AI capabilities.
- **Tiktoken**: Efficient tokenization for optimized performance with OpenAI models.
- **python-docx**: Create, edit, and extract data from Word documents.
- **Fitz (PyMuPDF)**: Tool for processing and extracting content from PDF files.

## Copyright

Memora Chatbot is developed by Team 2 CendekiAwan RevoU x AWS 2024. All materials featured are offered for individual or commercial information only and as such are offered on a 'as is' basis. Any other use of graphic material found on the site including mirroring or copying any part of the site is prohibited without the express written consent of Team 2 CendekiAwan RevoU or authorized representatives.
