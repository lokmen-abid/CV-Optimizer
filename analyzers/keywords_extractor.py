"""
Extracteur de mots-clés.

Ce module fournit :
- fonctions d'extraction de mots-clés utilisant KeyBERT, YAKE et spaCy
- fonctions utilitaires pour comparer CV vs JD et retourner listes avec scores

Toutes les opérations externes (KeyBERT, YAKE, spaCy)
sont optionnelles — le module gère leur absence et retourne des résultats partiels.

Format de sortie des extracteurs : List[Tuple[str, float]] (mot-clé, score), triée par score décroissante.

Conçu pour Python 3.8+.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Tentative d'import de config (silencieuse si non disponible)
try:
    from app import config  # type: ignore
except Exception:  # pragma: no cover - optional
    config = None


def _safe_lang(lang: Optional[str]) -> str:
    if lang is None:
        return "fr"
    return "fr" if str(lang).lower().startswith("fr") else "en"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


# Caches pour modèles (éviter rechargements coûteux)
_sentence_model_cache: Dict[str, object] = {}
_keybert_model_cache: Dict[str, object] = {}


def _get_sentence_model(model_name: str = 'all-MiniLM-L6-v2'):
    """Retourne une instance SentenceTransformer mise en cache (ou None si indisponible)."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        logger.debug("sentence-transformers non disponible (cache): %s", exc)
        return None

    if model_name in _sentence_model_cache:
        return _sentence_model_cache[model_name]

    try:
        model = SentenceTransformer(model_name)
        _sentence_model_cache[model_name] = model
        return model
    except Exception as exc:
        logger.exception("Impossible de charger SentenceTransformer %s: %s", model_name, exc)
        return None


def _get_keybert(model_name: str = 'all-MiniLM-L6-v2'):
    """Retourne une instance KeyBERT mise en cache (ou None si indisponible)."""
    try:
        from keybert import KeyBERT
    except Exception as exc:
        logger.debug("KeyBERT non disponible (cache): %s", exc)
        return None

    # use sentence-transformers encoder if available
    encoder = _get_sentence_model(model_name)
    cache_key = f"{model_name}_kb"
    if cache_key in _keybert_model_cache:
        return _keybert_model_cache[cache_key]

    try:
        kb = KeyBERT(model=encoder)
        _keybert_model_cache[cache_key] = kb
        return kb
    except Exception as exc:
        logger.exception("Impossible d'initialiser KeyBERT: %s", exc)
        return None


# ----------------- KeyBERT extractor ------------------------------------
def extract_keywords_keybert(text: str, top_n: int = 10, model: Optional[str] = None) -> List[Tuple[str, float]]:
    """Extrait des mots-clés avec KeyBERT en utilisant un cache pour le modèle.

    Args:
        text: texte source.
        top_n: nombre d'items à retourner.
        model: nom du modèle sentence-transformers utilisé par KeyBERT (ex: 'all-MiniLM-L6-v2').

    Returns:
        liste de tuples (keyword, score) ou [] si KeyBERT non disponible.
    """
    text = _normalize_text(text)
    if not text:
        return []

    model_name = model or (getattr(config, 'KEYBERT_MODEL', None) or 'all-MiniLM-L6-v2')
    kb = _get_keybert(model_name)
    if kb is None:
        return []

    try:
        results = kb.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n)
        return [(k, float(s)) for k, s in results]
    except Exception as exc:
        logger.exception("Erreur KeyBERT (cached): %s", exc)
        return []


# ----------------- YAKE extractor ---------------------------------------
def extract_keywords_yake(text: str, top_n: int = 10, lang: Optional[str] = None) -> List[Tuple[str, float]]:
    """Extrait des mots-clés avec YAKE.

    Args:
        text: texte source.
        top_n: nombre d'items à retourner.
        lang: 'fr' ou 'en' (si None, on déduira 'fr' par défaut).

    Returns:
        liste de tuples (keyword, score) où score plus petit => meilleur (YAKE) ; on renvoie 1/score pour cohéence.
    """
    text = _normalize_text(text)
    if not text:
        return []

    lang_code = _safe_lang(lang)

    try:
        import yake
    except Exception as exc:
        logger.debug("YAKE non disponible: %s", exc)
        return []

    try:
        kw_extractor = yake.KeywordExtractor(lan='fr' if lang_code == 'fr' else 'en', top=top_n)
        keywords = kw_extractor.extract_keywords(text)
        # YAKE returns (keyword, score) where lower score is better; convert to normalized score (1/score)
        out: List[Tuple[str, float]] = []
        for k, s in keywords:
            try:
                score = float(s)
                norm = 1.0 / score if score > 0 else 0.0
            except Exception:
                norm = 0.0
            out.append((k, norm))
        # sort by norm desc
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_n]
    except Exception as exc:
        logger.exception("Erreur YAKE: %s", exc)
        return []


# ----------------- spaCy extractor --------------------------------------
def extract_keywords_spacy(text: str, top_n: int = 10, lang: Optional[str] = None) -> List[Tuple[str, float]]:
    """Extrait des candidats keywords via spaCy (lemmes / noun_chunks + frequency).

    Args:
        text: texte source.
        top_n: nombre d'items à retourner.
        lang: 'fr' ou 'en'.

    Retourne:
        liste (keyword, score) où score = fréquence normalisée.
    """
    text = _normalize_text(text)
    if not text:
        return []

    lang_code = _safe_lang(lang)

    try:
        import spacy
    except Exception as exc:
        logger.debug("spaCy non disponible: %s", exc)
        return []

    try:
        model_name = 'fr_core_news_sm' if lang_code == 'fr' else 'en_core_web_sm'
        try:
            nlp = spacy.load(model_name)
        except Exception:
            # fallback: blank pipeline with tokenizer and sentencizer
            nlp = spacy.blank('fr' if lang_code == 'fr' else 'en')
            if 'sentencizer' not in nlp.pipe_names:
                nlp.add_pipe('sentencizer')

        doc = nlp(text)
        freq: Dict[str, int] = {}
        total = 0
        # Prefer noun chunks, fallback to lemmas of nouns/adjectives/verbs
        for chunk in doc.noun_chunks:
            k = chunk.text.strip().lower()
            freq[k] = freq.get(k, 0) + 1
            total += 1
        if total == 0:
            for token in doc:
                if token.is_stop or token.is_punct or not token.is_alpha:
                    continue
                if token.pos_ in ('NOUN', 'PROPN', 'ADJ', 'VERB'):
                    k = token.lemma_.lower()
                    freq[k] = freq.get(k, 0) + 1
                    total += 1
        if not freq:
            return []
        # build normalized scores
        items: List[Tuple[str, float]] = [(k, v / total) for k, v in freq.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_n]
    except Exception as exc:
        logger.exception("Erreur spaCy extraction: %s", exc)
        return []


# ----------------- Frequency extractor (fallback) -----------------------
def extract_keywords_frequency(text: str, top_n: int = 10, lang: Optional[str] = None) -> List[Tuple[str, float]]:
    """Extracteur fallback basé sur la fréquence brute des tokens (toujours disponible).

    Retourne une liste (token, score) où score est la fréquence normalisée.
    """
    txt = _normalize_text(text).lower()
    if not txt:
        return []
    tokens = [t for t in re.sub(r"[^\w\s]", " ", txt).split() if len(t) > 1]
    if not tokens:
        return []
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = sum(freq.values())
    items = [(k, v / total) for k, v in freq.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:top_n]


# ----------------- Aggregator / util ------------------------------------
def extract_keywords(text: str, top_n: int = 10, methods: Optional[List[str]] = None, lang: Optional[str] = None) -> Dict[str, List[Tuple[str, float]]]:
    """Extrait les mots-clés en combinant plusieurs méthodes.

    Args:
        text: texte source.
        top_n: nombre max par méthode.
        methods: liste parmi ['keybert','yake','spacy','frequency']; si None -> toutes disponibles.
        lang: 'fr' ou 'en'

    Returns:
        dict mapping method -> list[(keyword, score)].
    """
    if methods is None:
        methods = ['keybert', 'yake', 'spacy', 'frequency']

    res: Dict[str, List[Tuple[str, float]]] = {}

    if 'keybert' in methods:
        try:
            res['keybert'] = extract_keywords_keybert(text, top_n=top_n)
        except Exception:
            res['keybert'] = []
    if 'yake' in methods:
        try:
            res['yake'] = extract_keywords_yake(text, top_n=top_n, lang=lang)
        except Exception:
            res['yake'] = []
    if 'spacy' in methods:
        try:
            res['spacy'] = extract_keywords_spacy(text, top_n=top_n, lang=lang)
        except Exception:
            res['spacy'] = []
    if 'frequency' in methods:
        try:
            res['frequency'] = extract_keywords_frequency(text, top_n=top_n, lang=lang)
        except Exception:
            res['frequency'] = []

    return res


# ----------------- Helpers pour extraction spécifique ------------------
def extract_keywords_for_text(text: str, top_n: int = 10, methods: Optional[List[str]] = None, lang: Optional[str] = None) -> Dict[str, List[Tuple[str, float]]]:
    """Convenience wrapper: extrait keywords pour un texte donné (CV ou JD).

    Retourne le même format que `extract_keywords`.
    """
    return extract_keywords(text, top_n=top_n, methods=methods, lang=lang)


def extract_keywords_from_file(path: str, top_n: int = 10, methods: Optional[List[str]] = None, lang: Optional[str] = None) -> Dict[str, List[Tuple[str, float]]]:
    """Lit un fichier texte simple (.txt, .md) et extrait les keywords.

    Si le fichier n'est pas texte, essaie de réutiliser `parsers.cv_parser.parse_cv` si disponible
    pour extraire du texte depuis un PDF/DOCX.
    """
    if not path or not isinstance(path, str):
        return {}

    # Try to use parsers.parse_cv for richer extraction for common CV formats
    try:
        from parsers.cv_parser import parse_cv  # type: ignore
    except Exception:
        parse_cv = None

    text = ''
    try:
        if parse_cv is not None and path.lower().endswith(('.pdf', '.docx')):
            parsed = parse_cv(path)
            text = parsed.get('text', '') if isinstance(parsed, dict) else ''
        else:
            # read as plain text
            with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                text = fh.read()
    except Exception as exc:
        logger.debug('Impossible de lire/extraire le fichier %s: %s', path, exc)
        return {}

    return extract_keywords_for_text(text, top_n=top_n, methods=methods, lang=lang)


def extract_keywords_for_cv(source: str, from_file: bool = False, top_n: int = 10, methods: Optional[List[str]] = None) -> Dict[str, List[Tuple[str, float]]]:
    """Extract keywords specifically for a CV.

    Args:
        source: either the CV text or the path to a CV file if from_file=True.
        from_file: whether `source` should be treated as a file path.
        top_n: number of keywords per method.
        methods: methods to run (keybert, yake, spacy) or None for default.

    This function tries to detect language using `parsers.cv_parser.detect_language` when possible.
    """
    text = ''
    lang = None
    if from_file:
        # reuse extract_keywords_from_file (which may call parse_cv)
        try:
            res = extract_keywords_from_file(source, top_n=top_n, methods=methods, lang=None)
            # attempt to detect language using parsers
            try:
                from parsers.cv_parser import detect_language, extract_text  # type: ignore
                try:
                    raw = extract_text(source)
                    lang = detect_language(raw)
                except Exception:
                    lang = None
            except Exception:
                lang = None
            return res
        except Exception:
            return {}
    else:
        text = source or ''
        # detect language if possible
        try:
            from parsers.cv_parser import detect_language  # type: ignore
            lang = detect_language(text)
        except Exception:
            lang = None

    return extract_keywords_for_text(text, top_n=top_n, methods=methods, lang=lang)


# Export public minimal pour ce module (keywords only)
__all__ = [
    "extract_keywords_keybert",
    "extract_keywords_yake",
    "extract_keywords_spacy",
    "extract_keywords_frequency",
    "extract_keywords",
    "extract_keywords_for_text",
    "extract_keywords_from_file",
    "extract_keywords_for_cv",
]
