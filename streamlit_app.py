# streamlit_app.py

import streamlit as st
from topic_modeling import fetch_news_articles, preprocess, train_lda_model
import pyLDAvis.gensim_models
import pyLDAvis
import pickle

st.title("Topic Modeling on News Articles")
st.write("Analyze news articles to automatically detect underlying topics.")

# User input for search query
query = st.text_input("Enter a topic or keyword:", "machine learning")
num_articles = st.slider("Number of articles:", min_value=5, max_value=20, value=10)

# Button to fetch and process articles
if st.button("Fetch and Process Articles"):
    with st.spinner("Fetching articles..."):
        articles = fetch_news_articles(query, num_articles)
    
    with st.spinner("Training LDA Model..."):
        lda_model, coherence_score, dictionary, corpus = train_lda_model(articles)
    
    st.success("Model trained!")
    st.write(f"Coherence Score: {coherence_score}")
    
    # Display topics
    for idx, topic in lda_model.print_topics(num_words=5):
        st.write(f"**Topic {idx+1}:** {topic}")

    # Visualize topics
    with st.spinner("Generating topic visualization..."):
        lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(lda_display, 'lda_visualization.html')
        
        # Display as iframe
        st.markdown("<iframe src='lda_visualization.html' width='100%' height='600'></iframe>", unsafe_allow_html=True)
