import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# load Spacy Model
nlp = spacy.load("en_core_web_lg")

def preprocess_text(text):
     # Tokenize and lemmatize text using spaCy
     doc = nlp(text)
     tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct ]
     return " ".join(tokens)

def calculate_similarity(doc1, doc2):
     # Preprocess documents
     preprocess_doc1 = preprocess_text(doc1)
     preprocess_doc2 = preprocess_text(doc2)

     # Vectorize documents using TF-IDF
     vectorizer = TfidfVectorizer()
     tfidf_matrix = vectorizer.fit_transform([preprocess_doc1, preprocess_doc2])

     # Calculate cosine similarity between documents
     similarity_score = cosine_similarity(tfidf_matrix)[0, 1]
     return similarity_score

if __name__ == "__main__":
     # Example usage
     document1 = """ Natural language processing (NLP) is a field of computer science that focuses on 
        the interaction between computers and humans using natural language.
"""

     document2 = """ NLP involves the use of computer algorithms to process and understand natural language 
        data, enabling machines to interact with humans in a more human-like manner.
"""

# Calculate similarity score
similarity_score = calculate_similarity(document1, document2)
print(f"Similarity Score: {similarity_score:.2f}")
