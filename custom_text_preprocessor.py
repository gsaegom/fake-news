from sklearn.base import TransformerMixin, BaseEstimator

class CustomTextP
processor(BaseEstimator, TransformerMixin):
    def __init__(self, text_column=None, title_column=None):
        self.text_column = text_column
        self.title_column = title_column
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        if self.text_column:
            X_copy[self.text_column] = X_copy[self.text_column].apply(self._process_text)
        if self.title_column:
            X_copy[self.title_column] = X_copy[self.title_column].apply(self._process_text)
        return X_copy

    def fit(self, X, y=None):
        return self

    def _process_text(self, text):
        words = text.split()
        processed_words = []
        for word in words:
            if word.isupper():
                processed_words.append(word)
            else:
                processed_words.append(word.lower())
        return ' '.join(processed_words)

    def get_feature_names_out(self, input_features=None):
        return input_features
