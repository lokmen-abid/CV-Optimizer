"""
Similarity scorer pour comparer CV et Job Description (JD).

Fonctions:
- compute_tfidf_similarity(cv_text, jd_text, lang=None, top_n=20)
- compute_embedding_similarity(cv_text, jd_text, model_name='all-MiniLM-L6-v2')
- compare_cv_jd(cv_text, jd_text, top_n=20, lang=None)

Le module importe `analyzers.keywords_extractor` pour extraire les top terms de la JD
et les comparer au CV. Les dépendances (scikit-learn, sentence-transformers) sont optionnelles.
Des fallbacks sont fournis pour produire des scores même si ces dépendances sont absentes.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# importer keywords_extractor (extraction des top terms)
try:
    from analyzers.keywords_extractor import extract_keywords_for_text  # type: ignore
except Exception:
    extract_keywords_for_text = None  # type: ignore


# ----------------- helpers pour TF-IDF fallback -------------------------
def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    txt = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [t for t in txt.split() if t]
    return tokens


def _compute_tf(tokens: List[str]) -> Dict[str, float]:
    tf: Dict[str, float] = {}
    n = len(tokens)
    if n == 0:
        return tf
    for t in tokens:
        tf[t] = tf.get(t, 0.0) + 1.0
    # normalize by length
    for k in tf:
        tf[k] = tf[k] / n
    return tf


def _compute_idf(docs_tokens: List[List[str]]) -> Dict[str, float]:
    # idf = log((N+1)/(df+1)) + 1
    N = len(docs_tokens)
    df: Dict[str, int] = {}
    for tokens in docs_tokens:
        seen = set(tokens)
        for t in seen:
            df[t] = df.get(t, 0) + 1
    idf: Dict[str, float] = {}
    for t, v in df.items():
        idf[t] = math.log((N + 1) / (v + 1)) + 1.0
    return idf


def _build_vector(tf: Dict[str, float], idf: Dict[str, float], vocab: List[str]) -> List[float]:
    vec = []
    for term in vocab:
        vec.append(tf.get(term, 0.0) * idf.get(term, 1.0))
    return vec


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    sa = math.sqrt(sum(x * x for x in a))
    sb = math.sqrt(sum(y * y for y in b))
    if sa == 0 or sb == 0:
        return 0.0
    return num / (sa * sb)


# ----------------- compute_tfidf_similarity with fallback ----------------
def compute_tfidf_similarity(cv_text: str, jd_text: str, lang: Optional[str] = None, top_n: int = 20) -> Optional[Dict]:
    """Calcule la similarité TF-IDF (cosine) entre la JD et le CV.

    Tente d'utiliser scikit-learn si présent; sinon utilise un implémentation simple en pur Python.

    Retourne dict: { 'score': float, 'jd_top_terms': list[str], 'cv_top_terms': list[str] }
    ou None si aucune méthode n'a pu être exécutée (très rare).
    """
    # Try sklearn first
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        try:
            if not jd_text or not cv_text:
                return {"score": 0.0, "jd_top_terms": [], "cv_top_terms": []}
            stop_words = "english" if (lang and str(lang).lower().startswith('en')) else None
            vect = TfidfVectorizer(stop_words=stop_words, max_df=0.85, min_df=1)
            tfidf = vect.fit_transform([jd_text, cv_text])
            sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0])
            feature_names = vect.get_feature_names_out()
            jd_vec = tfidf[0].toarray().ravel()
            cv_vec = tfidf[1].toarray().ravel()
            jd_top_idx = jd_vec.argsort()[::-1][:top_n]
            cv_top_idx = cv_vec.argsort()[::-1][:top_n]
            jd_top = [feature_names[i] for i in jd_top_idx if jd_vec[i] > 0]
            cv_top = [feature_names[i] for i in cv_top_idx if cv_vec[i] > 0]
            logger.debug('compute_tfidf_similarity: used sklearn TF-IDF')
            return {"score": float(sim), "jd_top_terms": jd_top, "cv_top_terms": cv_top, "tfidf_method": "sklearn"}
        except Exception as exc:
            logger.debug("Erreur via sklearn TF-IDF, fallback: %s", exc)
            # fall through to pure-python fallback
    except Exception as exc:
        logger.debug("scikit-learn non disponible, utilisation du fallback TF-IDF: %s", exc)

    # Pure Python fallback
    try:
        if not jd_text or not cv_text:
            return {"score": 0.0, "jd_top_terms": [], "cv_top_terms": []}

        jd_tokens = _tokenize(jd_text)
        cv_tokens = _tokenize(cv_text)
        docs = [jd_tokens, cv_tokens]
        idf = _compute_idf(docs)
        jd_tf = _compute_tf(jd_tokens)
        cv_tf = _compute_tf(cv_tokens)

        # unified vocab order
        vocab = sorted(set(jd_tf.keys()) | set(cv_tf.keys()))
        jd_vec = _build_vector(jd_tf, idf, vocab)
        cv_vec = _build_vector(cv_tf, idf, vocab)
        sim = _cosine(jd_vec, cv_vec)

        # top terms by tf-idf value in JD and CV
        jd_scores = {term: (jd_tf.get(term, 0.0) * idf.get(term, 1.0)) for term in vocab}
        cv_scores = {term: (cv_tf.get(term, 0.0) * idf.get(term, 1.0)) for term in vocab}
        jd_top = [k for k, _v in sorted(jd_scores.items(), key=lambda x: x[1], reverse=True)[:top_n] if jd_scores[k] > 0]
        cv_top = [k for k, _v in sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)[:top_n] if cv_scores[k] > 0]

        logger.debug('compute_tfidf_similarity: used fallback pure-Python TF-IDF')
        return {"score": float(sim), "jd_top_terms": jd_top, "cv_top_terms": cv_top, "tfidf_method": "fallback"}
    except Exception as exc:
        logger.exception("Fallback TF-IDF failed: %s", exc)
        return None


# ----------------- compute_embedding_similarity with fallbacks ---------
def compute_embedding_similarity(cv_text: str, jd_text: str, model_name: str = "all-MiniLM-L6-v2") -> Optional[Dict[str, Any]]:
    """Calcule la similarité entre embeddings.

    Priorité:
    1. sentence-transformers
    2. spaCy doc.vector (si modèle possède vecteurs)
    3. fallback token overlap (Jaccard)

    Retourne un dict {'score': float, 'method': str} ou None si échec.
    """
    embed_method = None
    # 1) sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        try:
            model = SentenceTransformer(model_name)
            emb = model.encode([jd_text, cv_text], convert_to_numpy=True)
            a, b = emb[0], emb[1]
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                return {"score": 0.0, "method": "sentence-transformers"}
            sim = float((a @ b) / denom)
            logger.debug('compute_embedding_similarity: used sentence-transformers')
            embed_method = 'sentence-transformers'
            return {"score": max(0.0, sim), "method": embed_method}
        except Exception as exc:
            logger.debug('Erreur sentence-transformers encode: %s', exc)
    except Exception as exc:
        logger.debug('sentence-transformers non disponible: %s', exc)

    # 2) spaCy doc.vector if available
    try:
        import spacy
        try:
            # try to load a small model; fallback to blank
            try:
                nlp = spacy.load('fr_core_news_sm')
            except Exception:
                try:
                    nlp = spacy.load('en_core_web_sm')
                except Exception:
                    nlp = spacy.blank('fr') if (jd_text and re.search(r"\b(une|le|la|de|des)\b", jd_text.lower())) else spacy.blank('en')
            doc_j = nlp(jd_text)
            doc_c = nlp(cv_text)
            # spaCy returns meaningful vector only if model has vectors
            if hasattr(doc_j, 'vector') and hasattr(doc_c, 'vector') and len(doc_j.vector) and len(doc_c.vector):
                import numpy as np
                a = doc_j.vector
                b = doc_c.vector
                denom = (np.linalg.norm(a) * np.linalg.norm(b))
                if denom == 0:
                    return {"score": 0.0, "method": "spacy-vector"}
                sim = float((a @ b) / denom)
                logger.debug('compute_embedding_similarity: used spaCy doc.vector')
                embed_method = 'spacy-vector'
                return {"score": max(0.0, sim), "method": embed_method}
        except Exception as exc:
            logger.debug('spaCy vector fallback failed: %s', exc)
    except Exception as exc:
        logger.debug('spaCy non disponible pour fallback embeddings: %s', exc)

    # 3) simple token overlap (Jaccard)
    try:
        s1 = set(_tokenize(jd_text))
        s2 = set(_tokenize(cv_text))
        if not s1 and not s2:
            return {"score": 0.0, "method": "jaccard"}
        inter = len(s1 & s2)
        union = len(s1 | s2)
        if union == 0:
            return {"score": 0.0, "method": "jaccard"}
        logger.debug('compute_embedding_similarity: used Jaccard token-overlap fallback')
        embed_method = 'jaccard'
        return {"score": inter / union, "method": embed_method}
    except Exception as exc:
        logger.exception('Fallback embedding (Jaccard) failed: %s', exc)
        return None


# ----------------- compare_cv_jd ---------------------------------------
def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", (text or "")).lower()


def compare_cv_jd(cv_text: str, jd_text: str, top_n: int = 20, lang: Optional[str] = None) -> Optional[Dict]:
    """Compare CV vs JD.

    Retourne dict:
      - tfidf_score: float or None
      - embed_score: float or None
      - jd_keywords: list[str] (top terms from keywords_extractor or TF-IDF)
      - present_terms: list[str]
      - missing_terms: list[str]

    Utilise `extract_keywords_for_text` si disponible pour obtenir top keywords JD.
    """
    if not jd_text or not cv_text:
        return None

    result: Dict = {}

    # TF-IDF
    tfidf = compute_tfidf_similarity(cv_text, jd_text, lang=lang, top_n=top_n)
    result['tfidf_score'] = tfidf.get('score') if isinstance(tfidf, dict) else None
    result['tfidf_method'] = tfidf.get('tfidf_method') if isinstance(tfidf, dict) else None

    # Embeddings
    embed_res = compute_embedding_similarity(cv_text, jd_text)
    if isinstance(embed_res, dict):
        result['embed_score'] = embed_res.get('score')
        result['embed_method'] = embed_res.get('method')
    else:
        result['embed_score'] = embed_res
        result['embed_method'] = None

    # JD top keywords: prefer keywords_extractor output
    jd_keywords: List[str] = []
    try:
        if extract_keywords_for_text is not None:
            ke = extract_keywords_for_text(jd_text, top_n=top_n, methods=['spacy', 'yake', 'keybert'], lang=lang)
            # merge methods: take keys from YAKE then KeyBERT then spaCy by availability
            merged: List[str] = []
            for method in ('yake', 'keybert', 'spacy'):
                vals = ke.get(method) or []
                for k, _s in vals:
                    if k not in merged:
                        merged.append(k)
            jd_keywords = merged[:top_n]
    except Exception as exc:
        logger.debug('Erreur extraction keywords JD via keywords_extractor: %s', exc)

    # fallback: use tfidf top terms
    if not jd_keywords and isinstance(tfidf, dict):
        jd_keywords = tfidf.get('jd_top_terms', [])[:top_n]

    result['jd_keywords'] = jd_keywords

    # match present / missing
    norm_cv = _normalize_for_match(cv_text)
    present = []
    missing = []
    for term in jd_keywords:
        t = re.sub(r"[^\w\s]", " ", term).lower().strip()
        if re.search(rf"\b{re.escape(t)}\b", norm_cv):
            present.append(term)
        else:
            missing.append(term)

    result['present_terms'] = present
    result['missing_terms'] = missing

    return result


__all__ = [
    'compute_tfidf_similarity',
    'compute_embedding_similarity',
    'compare_cv_jd',
]
