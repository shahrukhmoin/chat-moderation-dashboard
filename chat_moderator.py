
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob

# Load or train model components
@st.cache_resource
def load_model():
    # Training sample data
    data = {
        "text": [
            "You're such a loser, no one likes you.", "Why don't you just disappear?",
            "Get lost, you're worthless.", "Watch your back tomorrow.",
            "I'll find you and deal with you.", "Great job on your project!",
            "See you at the library later.", "Thanks for your help today."
        ],
        "label": ["cyberbullying", "cyberbullying", "cyberbullying",
                  "threat", "threat",
                  "neutral", "neutral", "neutral"]
    }
    df = pd.DataFrame(data)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    model = MultinomialNB()
    model.fit(X, y)
    return model, vectorizer

model, vectorizer = load_model()

# GUI
st.title("📚 AI Chat Moderator for Schools")
st.write("Detect cyberbullying, threats, and sentiment in real-time student chat")

user_role = st.selectbox("Select your role:", ["Student", "Admin"])
user_input = st.text_area("Enter chat message:")

if st.button("Analyze Message") and user_input:
    vect_input = vectorizer.transform([user_input])
    prediction = model.predict(vect_input)[0]

    # Sentiment
    sentiment = TextBlob(user_input).sentiment.polarity
    sentiment_label = "Positive" if sentiment > 0 else ("Negative" if sentiment < 0 else "Neutral")

    # Output
    st.subheader("🔍 Analysis Result")
    st.write(f"**Detected Category:** `{prediction}`")
    st.write(f"**Sentiment Analysis:** `{sentiment_label}`")

    # Admin view
    if user_role == "Admin":
        st.success("This message has been logged for admin review.")
        st.dataframe(pd.DataFrame({"Message": [user_input], "Category": [prediction], "Sentiment": [sentiment_label]}))
    else:
        if prediction in ["cyberbullying", "threat"]:
            st.error("⚠️ Warning: This message may violate community guidelines.")
        else:
            st.success("✅ Message is clean.")
