# topic_modeling.py

# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim import corpora
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import requests
import json

from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec, CoherenceModel

# Download NLTK resources if not already present
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt_tab')

# Set up stopwords, stemmer, and lemmatizer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Fetch news articles from News API
API_KEY = '019c362ec11d44fb91aa17a4b10c2d86'

def fetch_news_articles(query, num_articles=5):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'pageSize': num_articles,
        'apiKey': API_KEY,
        'language': 'en'
    }
    response = requests.get(url, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        
        # Safely check if the 'articles' key exists in the response
        if 'articles' in data:
            articles = [article['description'] for article in data['articles'] if article['description']]
            return articles
        else:
            print("No articles found in the response.")
            return []
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

# Preprocess the text
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Train the LDA model
def train_lda_model(articles, num_topics=3):
    processed_articles = [preprocess(article) for article in articles]
    
    # Debugging: Print processed articles to check their content
    print("Processed Articles:", processed_articles)
    
    # Remove any empty articles from the list
    processed_articles = [article for article in processed_articles if article]
    
    if not processed_articles:
        print("No valid articles after preprocessing. Cannot train LDA model.")
        return None, None, None, None
    
    dictionary = corpora.Dictionary(processed_articles)
    corpus = [dictionary.doc2bow(article) for article in processed_articles]
    
    # Debugging: Print corpus to check its content
    print("Corpus:", corpus)
    
    # Check if the corpus is empty before passing to LDA model
    if not corpus:
        print("Corpus is empty. Cannot train LDA model.")
        return None, None, None, None
    
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_articles, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model_lda.get_coherence()
    
    return lda_model, coherence_score, dictionary, corpus


# Example of running the model
if __name__ == "__main__":
    query = "machine learning"
    articles = fetch_news_articles(query)
    lda_model, coherence_score, dictionary, corpus = train_lda_model(articles)

    print("Coherence Score:", coherence_score)
    for idx, topic in lda_model.print_topics(num_words=5):
        print(f"Topic {idx+1}: {topic}")
    
    # Visualize the topics
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda_visualization.html')
