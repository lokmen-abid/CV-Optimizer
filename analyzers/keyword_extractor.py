from __future__ import annotations

import logging
import re
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Fonctions publiques

def available_methods() -> List[str]:
    """Retourne la liste des méthodes d'extraction disponibles dans l'environnement.

    Exemples de méthodes: 'keybert', 'yake', 'tfidf'
    """
    methods = []
    try:
        import keybert  # type: ignore

        methods.append("keybert")
    except Exception:
        logger.debug("KeyBERT non disponible.")

    try:
        import yake  # type: ignore

        methods.append("yake")
    except Exception:
        logger.debug("YAKE non disponible.")

    try:
        import sklearn  # type: ignore

        methods.append("tfidf")
    except Exception:
        logger.debug("scikit-learn non disponible pour TF-IDF.")

    return methods


def _simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    # retirer ponctuation basique
    text = re.sub(r"[^\w\s\-]", " ", text)
    tokens = [t.strip() for t in re.split(r"\s+", text) if t.strip()]
    return tokens


def extract_keywords(text: str, top_n: int = 10, methods: Optional[List[str]] = None, lang: str = "en") -> List[Tuple[str, float]]:
    """Extrait des mots-clés depuis `text`.

    Priority: methods list (ex: ['keybert','yake','tfidf']). Si methods None, on essaie KeyBERT->YAKE->TF-IDF.

    Retourne une liste de tuples (keyword, score) triée par score décroissant.
    En cas d'absence totale de dépendances, retourne [] et loggue.
    """
    if not text or not text.strip():
        return []

    if methods is None:
        methods = ["keybert", "yake", "tfidf"]

    # Try KeyBERT
    if "keybert" in methods:
        try:
            from keybert import KeyBERT  # type: ignore

            kw_model = KeyBERT()
            res = kw_model.extract_keywords(text, top_n=top_n)
            # KeyBERT retourne list[(phrase, score)]
            logger.debug("Keywords extracted with KeyBERT: %d", len(res))
            return [(k, float(s)) for k, s in res]
        except Exception as exc:
            logger.debug("KeyBERT extraction failed: %s", exc)

    # Try YAKE
    if "yake" in methods:
        try:
            import yake  # type: ignore

            kw_extractor = yake.KeywordExtractor(lan=lang if lang in ("en", "fr") else "en", top=top_n)
            res = kw_extractor.extract_keywords(text)
            # YAKE retourne list[(phrase, score)] where lower is better -> invert score
            kws = [(k, 1.0 / (s + 1e-12)) for k, s in res]
            kws_sorted = sorted(kws, key=lambda x: x[1], reverse=True)
            logger.debug("Keywords extracted with YAKE: %d", len(kws_sorted))
            return kws_sorted
        except Exception as exc:
            logger.debug("YAKE extraction failed: %s", exc)

    # Fallback TF-IDF simple: top tokens by TF (local)
    if "tfidf" in methods:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            import numpy as np

            vect = TfidfVectorizer(max_features=top_n, stop_words='english' if lang == 'en' else None)
            X = vect.fit_transform([text])
            scores = X.toarray().sum(axis=0)
            terms = vect.get_feature_names_out()
            pairs = list(zip(terms, scores))
            pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_n]
            logger.debug("Keywords extracted with TF-IDF fallback: %d", len(pairs_sorted))
            return [(k, float(s)) for k, s in pairs_sorted]
        except Exception as exc:
            logger.debug("TF-IDF fallback failed: %s", exc)

    logger.warning("Aucune méthode d'extraction de mots-clés disponible")
    return []
