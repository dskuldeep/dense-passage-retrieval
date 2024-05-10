import os
import re
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_text_file(text_file_path):
    """Extract text from a text file."""
    with open(text_file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def preprocess_text(text):
    """Preprocess text by tokenizing, removing stopwords, and stemming."""
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalnum()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return tokens

def preprocess_data(directory):
    """Preprocess data by extracting text from files and tokenizing."""
    file_texts = {}
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_name.endswith('.txt'):
            text = extract_text_from_text_file(file_path)
        else:
            continue  # Skip files that are not PDF or text files
        file_texts[file_name] = {'original_text': text, 'preprocessed_text': preprocess_text(text)}
    return file_texts

# Example usage:
directory = 'cache'
preprocessed_data = preprocess_data(directory)
print(preprocessed_data)

from sentence_transformers import SentenceTransformer
import numpy as np

def generate_embeddings_with_sentence_transformers(model_name, texts):
    """Generate embeddings for a list of texts using Sentence Transformers."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings

# Example usage:
model_name = 'paraphrase-distilroberta-base-v1'  # You can choose a different model
texts = [' '.join(data['preprocessed_text']) for data in preprocessed_data.values()]  # Convert token lists back to strings
embeddings = generate_embeddings_with_sentence_transformers(model_name, texts)

# Save the embeddings, passages, and metadata for later use
np.save('embeddings.npy', embeddings)
np.save('file_names.npy', list(preprocessed_data.keys()))
