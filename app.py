import streamlit as st
import pickle
from remove_abbreviation import remove_abb
from nltk.stem import PorterStemmer
import spacy
nlp=spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

def TextPreprocessing(row):
    """
        It preprocess the given `Row of any Dataframe` or any Text.
        It provides tokenized review
    """
    remove_abb(row)  # expanding abbreviation, like, I don't --> I do not

    doc = nlp(row)  # tokenization using `spacy`

    tokens = []
    for token in doc:
        if not (token.like_url | token.is_stop | token.is_punct):
            tokens.append(token)

    tokens = [token.lemma_ for token in tokens]  # lemmatization

    tokens = [stemmer.stem(token) for token in tokens]  # stemming

    return tokens


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Movie Review Sentiment Analysis")

input_review = st.text_area("Enter the Review")

if st.button('Predict'):

    # 1. preprocess
    transformed_review = " ".join(TextPreprocessing(input_review))
    # 2. vectorize
    vector_input = tfidf.transform([transformed_review])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Positive")
    else:
        st.header("Negative")


