import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class BiasedTokenRemover(BaseEstimator, TransformerMixin):
    def __init__(self, text_column, title_column, threshold=0.95, ngram_range=(1, 3)):
        self.text_column = text_column
        self.title_column = title_column
        self.threshold = threshold
        self.biased_tokens = set()
        self.unbiased_tokens = set()
        self.ngram_range = ngram_range

    def fit(self, X, y=None):
        combined_texts = X[self.text_column] + ' ' + X[self.title_column]

        vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)
        vectorizer.fit(combined_texts)

        fake_texts = X[y == 0][self.text_column]
        true_texts = X[y == 1][self.text_column]

        fake_tfidf = vectorizer.transform(fake_texts)
        true_tfidf = vectorizer.transform(true_texts)

        fake_token_counts = fake_tfidf.sum(axis=0)
        true_token_counts = true_tfidf.sum(axis=0)

        fake_ratio = fake_token_counts / (fake_token_counts + true_token_counts + 10e-8)
        true_ratio = true_token_counts / (fake_token_counts + true_token_counts + 10e-8)

        self.biased_tokens = {token.lower() for token, fr, tr in zip(vectorizer.get_feature_names_out(), fake_ratio.A1, true_ratio.A1)
                              if fr >= self.threshold or tr >= self.threshold}
        self.unbiased_tokens = {token.lower() for token, fr, tr in zip(vectorizer.get_feature_names_out(), fake_ratio.A1, true_ratio.A1)
                                if fr < self.threshold and tr < self.threshold}
        return self

    def _preprocess_text(self, text):
        text = re.sub(r"[.']", "", text)
        return text

    def _remove_biased_tokens(self, text):
        text = self._preprocess_text(text)
        
        tokens = re.findall(r'\b\w+\b', text)
        
        return ' '.join([token for token in tokens if token.lower() not in self.biased_tokens])

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.text_column] = X_copy[self.text_column].apply(self._remove_biased_tokens)
        X_copy[self.title_column] = X_copy[self.title_column].apply(self._remove_biased_tokens)
        return X_copy
