import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import os
import re
import time

# Profanity list (can be extended)
PROFANITY_LIST = ["idiot", "dumb", "stupid", "loser", "shut up", "worthless"]

# Initialize session state for spam detection and alerts
if 'message_history' not in st.session_state:
    st.session_state.message_history = []
if 'user_violations' not in st.session_state:
    st.session_state.user_violations = 0
if 'user_blocked' not in st.session_state:
    st.session_state.user_blocked = False

# Load or train model components
@st.cache_resource
def load_model():
    # Expanded training sample data
    data = {
        "text": [
            "You're such a loser, no one likes you.", "Why don't you just disappear?",
            "Get lost, you're worthless.", "You're an idiot and everyone hates you.",
            "Nobody wants you here.", "You're so annoying, just stop talking.",
            "Watch your back tomorrow.", "I'll find you and deal with you.",
            "You better be careful after class.", "I'm warning you for the last time.",
            "Great job on your project!", "See you at the library later.",
            "Thanks for your help today.", "You did amazing in the presentation!",
            "Let's meet to finish the group task.", "You're a kind person.",
            "That was a very helpful session.", "Looking forward to our next class.",
            "Can you please help me with the assignment?", "Have a great weekend!"
        ],
        "label": [
            "cyberbullying", "cyberbullying", "cyberbullying",
            "cyberbullying", "cyberbullying", "cyberbullying",
            "threat", "threat", "threat", "threat",
            "neutral", "neutral", "neutral", "neutral",
            "neutral", "neutral", "neutral", "neutral",
            "neutral", "neutral"
        ]
    }
    df = pd.DataFrame(data)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    model = MultinomialNB()
    model.fit(X, y)
    return model, vectorizer

def contains_profanity(text):
    text_lower = text.lower()
    return any(bad_word in text_lower for bad_word in PROFANITY_LIST)


def is_spam(text):
    current_time = time.time()
    st.session_state.message_history = [msg for msg in st.session_state.message_history if current_time - msg['time']  30]
    recent_count = sum(1 for msg in st.session_state.message_history if msg['text'] == text)
    st.session_state.message_history.append({text text, time current_time})
    return recent_count = 2

model, vectorizer = load_model()

# GUI
st.title( AI Chat Moderator for Schools)
st.write(Detect cyberbullying, threats, profanity, spam, and sentiment in real-time student chat)

user_role = st.selectbox(Select your role, [Student, Admin])
user_input = st.text_area(Enter chat message)

# Show badge for violations
st.markdown(f###  Violations Count `{st.session_state.user_violations}`)

# Auto-block if violations exceed limit
if st.session_state.user_violations = 3
    st.session_state.user_blocked = True
    st.error(You have been temporarily blocked due to repeated violations.)

if st.button(Analyze Message) and user_input and not st.session_state.user_blocked
    vect_input = vectorizer.transform([user_input])
    proba = model.predict_proba(vect_input)[0]
    confidence = max(proba)
    prediction = model.predict(vect_input)[0]

    sentiment = TextBlob(user_input).sentiment.polarity
    sentiment_label = Positive if sentiment  0 else (Negative if sentiment  0 else Neutral)

    profanity_flag = contains_profanity(user_input)
    spam_flag = is_spam(user_input)

    violation_detected = False
    if prediction in [cyberbullying, threat] and confidence = 0.6
        violation_detected = True
    if profanity_flag or spam_flag
        violation_detected = True

    if violation_detected
        st.session_state.user_violations += 1

    st.subheader( Analysis Result)
    st.write(fDetected Category `{prediction}`)
    st.write(f"Confidence: {confidence:.2f}")
    st.write(fSentiment Analysis `{sentiment_label}`)
    if profanity_flag
        st.warning(Profanity Detected in this message.)
    if spam_flag
        st.warning( This message may be spam or repeated excessively.)
    if st.session_state.user_violations = 3
        st.error( Alert This user has been flagged 3 times for policy violations and is now blocked.)

    if user_role == Admin
        st.success(This message has been logged for admin review.)
        log_entry = pd.DataFrame({
            Message [user_input],
            Category [prediction],
            Sentiment [sentiment_label],
            Profanity [Yes if profanity_flag else No],
            Spam [Yes if spam_flag else No],
            Violations [st.session_state.user_violations]
        })

        if os.path.exists(chat_logs.csv)
            log_entry.to_csv(chat_logs.csv, mode='a', header=False, index=False)
        else
            log_entry.to_csv(chat_logs.csv, index=False)

        st.dataframe(log_entry)
    else
        if violation_detected
            st.warning( Please revise your message to follow community guidelines.)
        else
            st.success( Message is likely safe.)
