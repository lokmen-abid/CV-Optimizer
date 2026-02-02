from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


def load_spacy_model(lang: str = "en", prefer_full: bool = True):
    """Charge un modèle spaCy.

    Tentative de charger un modèle "full" (md/lg) si prefer_full True.
    Si indisponible, renvoie un pipeline blank pour tokenisation/sentencizer.

    Retourne l'objet nlp.
    """
    try:
        import spacy  # type: ignore
    except Exception as exc:
        logger.debug("spaCy non installé: %s", exc)
        return None

    model_candidates = []
    if lang.startswith("fr"):
        model_candidates = ["fr_core_news_md", "fr_core_news_sm", "fr_core_news_lg"]
    else:
        model_candidates = ["en_core_web_md", "en_core_web_sm", "en_core_web_lg"]

    if prefer_full:
        for m in model_candidates:
            try:
                nlp = spacy.load(m)
                logger.debug("Loaded spaCy model: %s", m)
                return nlp
            except Exception:
                continue

    # fallback: blank pipeline
    try:
        nlp = spacy.blank("fr" if lang.startswith("fr") else "en")
        # ensure sentencizer exists
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        logger.debug("Loaded blank spaCy pipeline for lang=%s", lang)
        return nlp
    except Exception as exc:
        logger.debug("Impossible de charger un pipeline spaCy: %s", exc)
        return None


def get_noun_chunks(text: str, nlp=None) -> List[str]:
    """Retourne une liste de noun chunks depuis `text`.

    Si `nlp` n'est pas fourni, tente de charger un modèle via `load_spacy_model`.
    Si la dépendance spaCy n'est pas disponible, retourne [] sans lever.
    """
    if not text:
        return []

    if nlp is None:
        nlp = load_spacy_model()

    if not nlp:
        return []

    try:
        doc = nlp(text)
        chunks = set()
        # some pipelines may not have dependency parse: guard
        if hasattr(doc, "noun_chunks"):
            for chunk in doc.noun_chunks:
                chunks.add(chunk.text.strip())
        else:
            # fallback: use noun phrases via POS heuristics (simple)
            for token in doc:
                if token.pos_ in ("NOUN", "PROPN"):
                    chunks.add(token.text.strip())
        return sorted(chunks)
    except Exception as exc:
        logger.debug("Erreur spaCy noun_chunks extraction: %s", exc)
        return []
