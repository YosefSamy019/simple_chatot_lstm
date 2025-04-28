import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import object_cache
import re
import spacy
import numpy as np
from datetime import datetime

# Constants
MAX_LEN = 12
tokenizer, le, tags_answers, model, nlp = None, None, None, None, None

@st.cache_data
def load():
    # Load cached objects and resources
    tokenizer = object_cache.loadObject('tokenizer')
    le = object_cache.loadObject('le')
    tags_answers = object_cache.loadObject('tags_answers')
    model = load_model('cache/chatbot_model.h5')
    nlp = spacy.load('en_core_web_sm')
    return tokenizer, le, tags_answers, model, nlp

def predict(msg):
    # Preprocess input message
    msg = str(msg)
    pat_char = re.compile(r'[^A-Za-z\s]')
    msg = re.sub(pat_char, ' ', msg.lower())
    msg = ' '.join([token.lemma_ for token in nlp(msg)])

    # Tokenize and pad the input
    sequences = tokenizer.texts_to_sequences([msg])
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=MAX_LEN)

    # Predict the tag
    y_hat = model.predict(padded_sequences)
    tag_index = np.argmax(y_hat)
    tag = le.inverse_transform([tag_index])[0]
    response = np.random.choice(tags_answers[tag])
    return response

def app():
    global tokenizer, le, tags_answers, model, nlp

    st.set_page_config(
        page_title="AI Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 AI Chatbot")
    st.markdown("Youssef Samy")
    st.markdown("🔗 The journey of building an intelligent chatbot combines creativity, data preprocessing, and cutting-edge technology. Through a mix of data exploration, NLP techniques, and deep learning models, ")

    st.markdown("Welcome! Ask me anything or type a message below to start chatting.")

    # Load resources with spinner
    with st.spinner("Loading chatbot resources..."):
        tokenizer, le, tags_answers, model, nlp = load()

    # Initialize chat messages
    if "msg" not in st.session_state:
        st.session_state["msg"] = []

    # User input field
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Append user message and bot response
        st.session_state["msg"].append((0, user_input))
        bot_response = predict(user_input)
        st.session_state["msg"].append((1, bot_response))

    # Display chat messages
    for sender, message in st.session_state["msg"]:
        if sender == 0:
            with st.chat_message("👱"):
                st.write( f"{message}")
        else:
            with st.chat_message("🤖"):
                st.write(f"{message}")

    # Sidebar for additional options
    with st.sidebar:
        st.header("Settings")
        if st.button("Clear Chat"):
            st.session_state["msg"] = []
            st.rerun()

if __name__ == "__main__":
    app()