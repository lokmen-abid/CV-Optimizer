from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _tokenize_simple(text: str) -> List[str]:
    if not text:
        return []
    txt = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in txt.split() if t]


def _detect_language_from_text(jd_text: str, cv_text: Optional[str] = None) -> str:
    """Heuristique simple pour détecter la langue (fr/en) à partir du texte fourni."""
    sample = (jd_text or '')[:2000].lower()
    french_indicators = ["profil", "compétences", "expérience", "formation", "langues", "résumé", "le", "la", "des"]
    en_indicators = ["the", "and", "of", "experience", "skills"]
    fr_score = sum(1 for w in french_indicators if w in sample)
    en_score = sum(1 for w in en_indicators if w in sample)
    if fr_score >= en_score and fr_score > 0:
        return 'fr'
    return 'en'


# --- Use centralized get_model from analyzers.embeddings -----------------
try:
    from analyzers.embeddings import get_model  # type: ignore
except Exception:
    # If embeddings module missing, we'll fallback gracefully
    def get_model(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        return None


def compute_tfidf_similarity(cv_text: str, jd_text: str, lang: Optional[str] = None, top_n: int = 20) -> Optional[Dict]:
    """Calcule similarité TF-IDF et extrait top terms pour JD et CV.

    Supporte le français et l'anglais via `lang` ou heuristique.
    Retourne dict: {'score': float, 'jd_top_terms': list, 'cv_top_terms': list, 'tfidf_method': str}
    """
    cv_text = (cv_text or '').strip()
    jd_text = (jd_text or '').strip()
    if not cv_text or not jd_text:
        return {'score': 0.0, 'jd_top_terms': [], 'cv_top_terms': [], 'tfidf_method': 'none'}

    if lang is None:
        try:
            lang = _detect_language_from_text(jd_text, cv_text)
        except Exception:
            lang = 'en'

    # Essayer sklearn
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        stop_words = 'english' if (lang and str(lang).lower().startswith('en')) else 'french'
        vect = TfidfVectorizer(stop_words=stop_words, max_df=0.95, min_df=1, ngram_range=(1,2))
        tfidf = vect.fit_transform([jd_text, cv_text])
        feature_names = vect.get_feature_names_out()
        if tfidf.shape[1] == 0:
            return {'score': 0.0, 'jd_top_terms': [], 'cv_top_terms': [], 'tfidf_method': 'sklearn'}
        sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0])
        jd_vec = tfidf[0].toarray().ravel()
        cv_vec = tfidf[1].toarray().ravel()
        # numpy argsort may include zeros; we filter later
        jd_top_idx = np.argsort(jd_vec)[::-1][:top_n]
        cv_top_idx = np.argsort(cv_vec)[::-1][:top_n]
        jd_top = [feature_names[i] for i in jd_top_idx if jd_vec[i] > 0]
        cv_top = [feature_names[i] for i in cv_top_idx if cv_vec[i] > 0]
        return {'score': float(sim), 'jd_top_terms': jd_top, 'cv_top_terms': cv_top, 'tfidf_method': 'sklearn'}
    except Exception as exc:
        logger.debug('sklearn TF-IDF non disponible ou erreur: %s', exc)

    # Fallback simple: vectoriser par fréquences + idf approximatif
    try:
        jd_tokens = _tokenize_simple(jd_text)
        cv_tokens = _tokenize_simple(cv_text)
        if not jd_tokens or not cv_tokens:
            return {'score': 0.0, 'jd_top_terms': [], 'cv_top_terms': [], 'tfidf_method': 'fallback'}
        from math import log, sqrt
        docs = [set(jd_tokens), set(cv_tokens)]
        df = {}
        for s in docs:
            for t in s:
                df[t] = df.get(t, 0) + 1
        idf = {t: log((len(docs) + 1) / (df.get(t, 0) + 1)) + 1.0 for t in set(jd_tokens + cv_tokens)}

        def tf(tokens):
            from collections import Counter
            c = Counter(tokens)
            n = len(tokens)
            return {k: v / n for k, v in c.items()}

        jd_tf = tf(jd_tokens)
        cv_tf = tf(cv_tokens)
        vocab = sorted(set(list(jd_tf.keys()) + list(cv_tf.keys())))
        jd_vec = [jd_tf.get(w, 0.0) * idf.get(w, 1.0) for w in vocab]
        cv_vec = [cv_tf.get(w, 0.0) * idf.get(w, 1.0) for w in vocab]
        num = sum(a * b for a, b in zip(jd_vec, cv_vec))
        denom = (sqrt(sum(a * a for a in jd_vec)) * sqrt(sum(b * b for b in cv_vec)))
        score = float(num / denom) if denom != 0 else 0.0
        jd_scores = {w: jd_tf.get(w, 0.0) * idf.get(w, 1.0) for w in vocab}
        cv_scores = {w: cv_tf.get(w, 0.0) * idf.get(w, 1.0) for w in vocab}
        jd_top = [k for k, _ in sorted(jd_scores.items(), key=lambda x: x[1], reverse=True)[:top_n] if jd_scores[k] > 0]
        cv_top = [k for k, _ in sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)[:top_n] if cv_scores[k] > 0]
        return {'score': score, 'jd_top_terms': jd_top, 'cv_top_terms': cv_top, 'tfidf_method': 'fallback'}
    except Exception as exc:
        logger.exception('Fallback TF-IDF failed: %s', exc)
        return None


def keyword_overlap(cv_text: str, jd_text: str) -> Dict:
    """Calcule pourcentage d'overlap de mots entre JD et CV (word token set).

    """
    cv_tokens = set(_tokenize_simple(cv_text))
    jd_tokens = set(_tokenize_simple(jd_text))
    if not jd_tokens:
        return {'keyword_match': 0.0, 'matching_keywords': []}
    overlap = cv_tokens.intersection(jd_tokens)
    score = len(overlap) / len(jd_tokens) * 100.0
    return {'keyword_match': round(score, 1), 'matching_keywords': sorted(list(overlap))}


def compute_embedding_similarity(cv_text: str, jd_text: str, model_name: str = "all-MiniLM-L6-v2") -> Optional[Dict[str, float]]:
    """Calcule la similarité par embeddings entre la JD et le CV.

    Essaie d'utiliser `analyzers.embeddings` (caching, chunking) si disponible.
    Sinon, conserve les méthodes précédentes (sentence-transformers direct, spaCy vectors, Jaccard fallback).

    Retourne: {'score': float, 'method': str} ou None en cas d'erreur.
    """
    cv_text = (cv_text or '').strip()
    jd_text = (jd_text or '').strip()
    if not cv_text or not jd_text:
        return {'score': 0.0, 'method': 'none'}

    # Prefer using the embeddings manager (with cache & preload)
    try:
        from analyzers.embeddings import compute_embedding_similarity as _emb_comp  # type: ignore
        try:
            res = _emb_comp(cv_text, jd_text, model_name=model_name, use_cache=True)
            if isinstance(res, dict):
                return res
        except Exception as exc:
            logger.debug('analyzers.embeddings.compute_embedding_similarity failed: %s', exc)
    except Exception:
        # embeddings manager not available
        pass

    # 1) sentence-transformers (fallback)
    try:
        import numpy as np
        # use centralized get_model from analyzers.embeddings
        m = get_model(f"sentence-transformers/{model_name}" if not str(model_name).startswith("sentence-transformers") else model_name)
        if m is None:
            raise Exception("sentence-transformers model unavailable")
        emb = m.encode([jd_text, cv_text], convert_to_numpy=True)
        a, b = emb[0], emb[1]
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return {'score': 0.0, 'method': 'sentence-transformers'}
        sim = float((a @ b) / denom)
        return {'score': max(0.0, float(sim)), 'method': 'sentence-transformers'}
    except Exception as exc:
        logger.debug('sentence-transformers non disponible ou erreur (centralized loader): %s', exc)

    # 2) spaCy vector
    try:
        import spacy
        # choisir modèle par heuristique (si installé)
        lang = 'en' if re.search(r"\bthe\b|\band\b|\bof\b", jd_text.lower()) else 'fr'
        model_name_spacy = 'en_core_web_md' if lang == 'en' else 'fr_core_news_md'
        nlp = spacy.load(model_name_spacy)
        doc_j = nlp(jd_text)
        doc_c = nlp(cv_text)
        if hasattr(doc_j, 'vector') and hasattr(doc_c, 'vector') and len(doc_j.vector) and len(doc_c.vector):
            import numpy as np
            a = doc_j.vector
            b = doc_c.vector
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                return {'score': 0.0, 'method': 'spacy-vector'}
            sim = float((a @ b) / denom)
            return {'score': max(0.0, float(sim)), 'method': 'spacy-vector'}
    except Exception as exc:
        logger.debug('spaCy vector non disponible ou erreur: %s', exc)

    # 3) Jaccard fallback
    try:
        s1 = set(_tokenize_simple(jd_text))
        s2 = set(_tokenize_simple(cv_text))
        if not s1 and not s2:
            return {'score': 0.0, 'method': 'jaccard'}
        inter = len(s1 & s2)
        union = len(s1 | s2)
        if union == 0:
            return {'score': 0.0, 'method': 'jaccard'}
        return {'score': inter / union, 'method': 'jaccard'}
    except Exception as exc:
        logger.exception('Jaccard fallback failed: %s', exc)
        return None


__all__ = ['compute_tfidf_similarity', 'keyword_overlap', 'compute_embedding_similarity']
