from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer


def text_data_analysis(data, text_column):
    """Perform basic NLP analysis on a text column."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data[text_column].fillna(''))

    return tfidf_matrix
