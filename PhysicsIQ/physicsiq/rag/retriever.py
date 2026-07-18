"""retriever.py — picks the FreeCAD recipes relevant to the user's prompt.

Why RAG at all? A 2B model has NOT memorized the FreeCAD Python API.
But it is a decent pattern-copier. So per chat turn we paste the 2-3
most relevant known-good recipes (rag/snippets/*.md) into the system
prompt and the model adapts them instead of hallucinating API calls.

Why BM25 and not embeddings? BM25 is ~60 lines of pure Python, needs no
extra model in VRAM, and on a corpus of ~15 hand-written snippets it is
essentially as good. Swap this file for an embedding retriever later if
the corpus grows to hundreds of files — the interface is one function.
"""

import math
import os
import re

_SNIPPET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "snippets")
_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

# BM25 magic constants — the values every paper/library defaults to.
_K1 = 1.5
_B = 0.75


def _tokenize(text: str):
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class _Corpus:
    """Loads every snippet once and pre-computes BM25 statistics."""

    def __init__(self):
        self.docs = []       # [(filename, full_text, token_list)].
        self.df = {}         # token -> in how many docs it appears.
        for name in sorted(os.listdir(_SNIPPET_DIR)):
            if not name.endswith(".md"):
                continue
            with open(os.path.join(_SNIPPET_DIR, name), encoding="utf-8") as fh:
                text = fh.read()
            tokens = _tokenize(text)
            self.docs.append((name, text, tokens))
            for tok in set(tokens):
                self.df[tok] = self.df.get(tok, 0) + 1
        self.avg_len = (sum(len(t) for _, _, t in self.docs) /
                        max(1, len(self.docs)))

    def score(self, query_tokens, doc_tokens):
        # Textbook BM25: rare query words matching a doc score high,
        # long docs are slightly penalized so they don't win by bulk.
        n = len(self.docs)
        counts = {}
        for tok in doc_tokens:
            counts[tok] = counts.get(tok, 0) + 1
        s = 0.0
        for tok in query_tokens:
            tf = counts.get(tok, 0)
            if tf == 0:
                continue
            df = self.df.get(tok, 0)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            norm = tf * (_K1 + 1) / (
                tf + _K1 * (1 - _B + _B * len(doc_tokens) / self.avg_len))
            s += idf * norm
        return s


_corpus = None


def _get_corpus():
    global _corpus
    if _corpus is None:
        _corpus = _Corpus()  # lazy: only pay file-reading cost when used
    return _corpus


def retrieve(query: str, top_k: int = 3, char_budget: int = 3500) -> str:
    """Return the top_k most relevant snippets, concatenated, trimmed to
    a character budget so the context window never overflows."""
    corpus = _get_corpus()
    q = _tokenize(query)
    ranked = sorted(
        ((corpus.score(q, toks), name, text)
         for name, text, toks in corpus.docs),
        key=lambda t: t[0], reverse=True)

    picked, used = [], 0
    for score, _name, text in ranked[:top_k]:
        if score <= 0:
            break  # zero overlap with the query — pasting it would be noise
        if used + len(text) > char_budget:
            text = text[:char_budget - used]
        picked.append(text)
        used += len(text)
        if used >= char_budget:
            break
    return "\n\n".join(picked) if picked else "(no matching recipe)"
