import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk  # Import nltk

# Download NLTK resources (check for availability first)
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    st.write("Downloading NLTK 'stopwords'...")
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
except LookupError:
    st.write("Downloading NLTK 'wordnet'...")
    nltk.download('wordnet')  # Download wordnet here
    lemmatizer = WordNetLemmatizer()



# Load model and vectorizer
try:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError:
    st.error("Error: 'fake_news_model.pkl' or 'tfidf_vectorizer.pkl' not found.  Make sure these files are in the same directory as your script, or update the paths.")
    st.stop() # Stop if model files are missing.



# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit app UI
st.title("📰 Fake News Detection")

input_text = st.text_area("Enter a news article or paragraph:")

if st.button("Check"):
    if not input_text: #check if the input_text is empty
        st.warning("Please enter a news article or paragraph to check.")
    else:
        cleaned = preprocess(input_text)
        try: #wrap the prediction in a try-except block
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
            st.success(f"Prediction: {label}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
