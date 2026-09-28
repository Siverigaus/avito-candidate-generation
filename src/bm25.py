"""Разреженный BM25: матрица весов документов, скоринг запросов батчами."""
import numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

def _ident(x):
    return x

class BM25:
    def __init__(self, k1=1.2, b=0.75, vocabulary=None):
        self.k1, self.b = k1, b
        self.cv = CountVectorizer(analyzer=_ident, vocabulary=vocabulary)

    def fit(self, docs):
        tf = self.cv.fit_transform(docs).tocsr().astype(np.float32)
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel(); avg = dl.mean() if dl.mean() > 0 else 1
        denom_row = self.k1 * (1 - self.b + self.b * dl / avg)
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        d = tf.data
        tf.data = (self.idf[tf.indices] * d * (self.k1 + 1) / (d + denom_row[rows])).astype(np.float32)
        self.W = tf.T.tocsr()   # vocab x docs
        return self

    def query_matrix(self, qdocs):
        q = self.cv.transform(qdocs).tocsr().astype(np.float32)
        q.data[:] = 1.0
        return q

    def score(self, q):
        """q: (nq x vocab) -> dense (nq x ndocs)."""
        return (q @ self.W).toarray()
