import re
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import STOPWORDS, WordCloud
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')



def count_words(text: str) -> int:
    """
    Count the number of words in a given text.

    Args:
        text (str): Input text to count words.

    Returns:
        int: The number of words in the text.
    """
    return len(text.split(' '))


def count_uppercase_words(text: str) -> int:
    """
    Count the number of uppercase words in a given text.

    Args:
        text (str): Input text to check for uppercase words.

    Returns:
        int: The number of uppercase words in the text.
    """
    words = text.split()
    uppercase_count = sum(1 for word in words if word.isupper())
    return uppercase_count


def get_uppercase_word_ratio(text: str) -> float:
    """
    Calculate the ratio of uppercase words to the total number of words.

    Args:
        text (str): Input text to calculate the ratio.

    Returns:
        float: The ratio of uppercase words to total words.
    """
    return count_uppercase_words(text) / count_words(text)


def plot_word_cloud(column: pd.Series) -> None:
    """
    Generate and display a word cloud from a given text column.

    Args:
        column (pd.Series): Series containing the text data to generate the word cloud.

    Returns:
        None
    """
    comment_words = ''
    stopwords = set(STOPWORDS)

    for val in column:
        val = str(val)
        tokens = val.split()

        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += " ".join(tokens) + " "

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate(comment_words)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespaces and special characters.

    Args:
        text (str): Input text to clean.

    Returns:
        str: The cleaned text.
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text
def preprocess_text(text):
    words = text.split()
    words = [word for word in words if word.lower() not in ENGLISH_STOP_WORDS]
    return ' '.join(words)


def predict_proba(texts):
    processed_texts = [preprocess_text(t) for t in texts]
    df = pd.DataFrame({
        'title': [t.split('|||')[0] for t in processed_texts],
        'text': [t.split('|||')[1] for t in processed_texts],
        'n_words_title': [len(t.split('|||')[0].split()) for t in
                          processed_texts],
        'n_words_text': [len(t.split('|||')[1].split()) for t in
                         processed_texts],
        'n_capitalised_words_title': [count_uppercase_words(t.split('|||')[0])
                                      for t in processed_texts],
        'n_capitalised_words_text': [count_uppercase_words(t.split('|||')[1])
                                     for t in processed_texts],
        'ratio_capitalised_words_title': [count_uppercase_words(t.split('|||')
                                                                [0]) / len(
            t.split('|||')[0].split())
                                          for t in processed_texts],
        'ratio_capitalised_words_text': [
            count_uppercase_words(t.split('|||')[1])
            / len(t.split('|||')[1].split())
            for t in processed_texts]
    })
    transformed_data = model_pipeline.named_steps['preprocessor'].transform(df)
    return model_pipeline.named_steps['classifier'].predict_proba(
        transformed_data)

def get_ngrams(text_series, ngram_type=2):
    """
    Extracts the most common bigrams and trigrams from a pandas Series of text.
    
    Parameters:
    - text_series: Pandas Series containing text data.
    - ngram_range: Tuple specifying the range of n-grams to consider (e.g., (2, 3) for bigrams and trigrams).
    
    Returns:
    - A DataFrame containing n-grams and their counts.
    """
    stop_words = list(stopwords.words('english'))
    vectorizer = CountVectorizer(ngram_range=(ngram_type,ngram_type), stop_words=stop_words)
    X = vectorizer.fit_transform(text_series)
    ngrams = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1
    ngram_counts = pd.DataFrame({'ngram': ngrams, 'count': counts})
    return ngram_counts.sort_values(by='count', ascending=False)
