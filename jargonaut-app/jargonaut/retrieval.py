"""Step 3: rank a contract's chunks for one category with BM25.

The final setup of BM25_Retrieval_and_Evaluation.ipynb (Kiara, Ridheema), frozen
before their test run: query form B (category name + description from the CUAD
question), lowercase alphanumeric tokens, NLTK English stopwords removed except
the negations no/nor/not/without, Porter stemming, k1=1.5, b=0.75. One index per
contract, so IDF is computed over that contract's chunks only.
"""
import re
from functools import lru_cache

from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

from .config import BM25_B, BM25_K1, BM25_TOP_K

NLTK_STOPWORDS = '''i me my myself we our ours ourselves you your yours yourself yourselves he him his himself
she her hers herself it its itself they them their theirs themselves what which who whom this that these those
am is are was were be been being have has had having do does did doing a an the and but if or because as until
while of at by for with about against between into through during before after above below to from up down in
out on off over under again further then once here there when where why how all any both each few more most
other some such only own same so than too very s t can will just don should now d ll m o re ve y'''.split()
STOPWORDS = set(NLTK_STOPWORDS) - {"no", "nor", "not", "without"}
_stemmer = PorterStemmer()


@lru_cache(maxsize=None)
def stem(word):
    return _stemmer.stem(word)


def tokenize(text):
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


TEMPLATE = re.compile(r'related to "(.+?)" that should be reviewed by a lawyer\.\s*Details:\s*(.*)', re.S)


def query_form_b(question):
    """Category name + description, without the CUAD boilerplate around them."""
    m = TEMPLATE.search(question)
    if m is None:
        raise ValueError(f"Not a CUAD template question: {question[:80]}")
    return " ".join(m.groups())


class ChunkIndex:
    """BM25 over one contract's chunks."""

    def __init__(self, chunks, k1=BM25_K1, b=BM25_B):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks], k1=k1, b=b)

    def top_k(self, question, k=BM25_TOP_K):
        """[(chunk, score)] for the k best chunks, best first. Ties keep chunk order."""
        query = query_form_b(question)
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]
