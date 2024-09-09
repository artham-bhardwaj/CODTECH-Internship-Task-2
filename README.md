# Topic Modeling on News Articles

## Project Details

- **Name**: Artham Bhardwaj
- **Company**: CodTech IT Solutions
- **ID**: CT12DS1704
- **Domain**: DATA SCIENCE
- **Duration**: July 10, 2024 - September 10, 2024
- **Mentor**: Mr. Muzammil Ahmed

## Overview of the Project

### Project Title
**Topic Modeling on News Articles**

### Objective
The primary objective of this project is to automatically detect topics within a collection of news articles. By analyzing the text data, the project aims to identify and visualize common themes, which can aid in news summarization, trend analysis, and information retrieval.

### Key Activities
1. **Data Collection**: Fetch news articles using the Google News API based on a specific query.
2. **Text Preprocessing**: Clean and preprocess the text data by tokenization, stopword removal, stemming, and lemmatization.
3. **LDA Model Training**: Apply Latent Dirichlet Allocation (LDA) to identify topics within the text data.
4. **Word Embeddings with Word2Vec**: Train a Word2Vec model to learn vector representations of words for semantic analysis.
5. **Visualization**: Use `pyLDAvis` to visualize the topics and understand the relationships between them.
6. **Coherence Score Calculation**: Evaluate the quality of topics using coherence scores to ensure they are meaningful and interpretable.

### Technology Used
- **Programming Language**: Python 3.x
- **Libraries**: 
  - `nltk`
  - `requests`
  - `gensim`
  - `pyLDAvis`
  - `matplotlib`

### Dataset Used
- **Dataset**: News articles fetched from the Google News API.
- **Description**: The dataset includes news articles related to a specific query (e.g., 'machine learning') with descriptions extracted for topic modeling.

### Insights
- The LDA model identified several topics within the news articles, providing a clear understanding of the main themes covered.
- Word embeddings generated using Word2Vec captured semantic relationships between words, enhancing topic interpretation.
- Visualizations of topics and their relationships facilitated easy understanding of the key themes and their distribution.

### Result and Conclusion
The topic modeling project effectively identified and visualized key topics within the news articles. The LDA model and Word2Vec embeddings provided valuable insights into the main themes, and the coherence score indicated that the topics were coherent and interpretable. The visualizations generated offer a comprehensive view of the topics and their relationships, aiding in the analysis and summarization of news content.

### Output Visualizations

Below are some visualizations from the topic modeling analysis:

- **LDA Topics Visualization**

![image](https://github.com/user-attachments/assets/c29e4ae2-c42d-45be-8560-e3d6dbea281d)


- **Word2Vec Vocabulary and Similar Words**

 ![image](https://github.com/user-attachments/assets/f6f1858c-49c9-47ed-8456-7c8da95b636e)




- **Coherence Score**

 ![image](https://github.com/user-attachments/assets/9997319e-72ca-4cb0-93a4-f60ba4b375ed)

