"""
Parser de Job Description (JD) pour ATS-Optimizer.

Ce module fournit des fonctions pour :
- lire / extraire le texte d'une JD depuis une chaîne ou un fichier texte (.txt, .md),
- nettoyer et normaliser le texte (espaces, retours, caractères non imprimables),
- détecter la langue (fr/en) via heuristique simple.

Les fonctions sont defensives et documentées. Conçu pour Python 3.8+.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Tenter d'importer des utilitaires depuis parsers.cv_parser si présent
try:
    from .cv_parser import clean_text as _cv_clean_text, detect_language as _cv_detect_language  # type: ignore
except Exception:
    _cv_clean_text = None  # type: ignore
    _cv_detect_language = None  # type: ignore


def read_text_file(path: str, encoding: str = "utf-8") -> str:
    """Lit un fichier texte (.txt, .md) et retourne son contenu.

    Args:
        path: chemin vers le fichier texte.
        encoding: encodage utilisé pour la lecture (par défaut 'utf-8').

    Returns:
        Le contenu du fichier sous forme de chaîne.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
        RuntimeError: pour d'autres erreurs d'I/O.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    try:
        with open(path, "r", encoding=encoding, errors="replace") as fh:
            return fh.read()
    except Exception as exc:
        logger.exception("Erreur lecture fichier JD: %s", exc)
        raise RuntimeError("Impossible de lire le fichier JD") from exc


def clean_jd_text(text: str, collapse_newlines: bool = True) -> str:
    """Nettoie et normalise le texte d'une Job Description.

    Opérations réalisées:
    - normalisation des retours chariot et des espaces
    - suppression de caractères non imprimables
    - collapse des multiples newlines (optionnel)

    Cette fonction essaie de réutiliser `clean_text` depuis `parsers.cv_parser` si
    disponible, sinon applique une implémentation interne légère.

    Args:
        text: texte brut.
        collapse_newlines: si True, remplace 3+ newlines par 2 newlines.

    Returns:
        Texte nettoyé.
    """
    if not text:
        return ""

    # Si la fonction du module cv_parser est disponible, l'utiliser (consistance)
    if _cv_clean_text is not None:
        try:
            return _cv_clean_text(text, collapse_newlines=collapse_newlines)  # type: ignore
        except Exception as exc:
            logger.debug("_cv_clean_text a échoué, fallback local: %s", exc)

    # Implémentation locale
    txt = text.replace("\r\n", "\n").replace("\r", "\n")
    txt = "".join(ch for ch in txt if ch.isprintable() or ch == "\n")
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    if collapse_newlines:
        txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = txt.strip()
    return txt


def detect_language_jd(text: str) -> str:
    """Heuristique simple pour déterminer si la JD est en français ou anglais.

    Renvoie 'fr' ou 'en'. Si une fonction de détection existe dans `parsers.cv_parser`,
    on la réutilise.
    """
    if _cv_detect_language is not None:
        try:
            return _cv_detect_language(text)  # type: ignore
        except Exception as exc:
            logger.debug("_cv_detect_language a échoué, fallback local: %s", exc)

    sample = (text or "")[:1500].lower()
    french_indicators = [
        "description de poste",
        "profil",
        "compétences",
        "langues",
        "expérience",
        "poste",
        "missions",
        "contrat",
    ]
    fr_score = sum(1 for w in french_indicators if w in sample)
    return "fr" if fr_score >= 1 else "en"


def parse_jd(source: str, from_file: bool = False, do_clean: bool = True) -> Dict[str, Optional[str]]:
    """Orchestre l'extraction et le nettoyage d'une Job Description.

    Args:
        source: soit le texte brut de la JD, soit le chemin vers un fichier texte si
                `from_file=True`.
        from_file: si True, `source` est traité comme un chemin de fichier et lu.
        do_clean: si True, applique `clean_jd_text`.

    Returns:
        dict contenant au moins les clés: 'text' (texte total) et 'lang' (fr/en).

    Raises:
        Exception: si la lecture du fichier échoue (si from_file=True).
    """
    text = source
    if from_file:
        text = read_text_file(source)

    if do_clean:
        text = clean_jd_text(text)

    lang = detect_language_jd(text)

    return {"text": text, "lang": lang}


__all__ = [
    "read_text_file",
    "clean_jd_text",
    "detect_language_jd",
    "parse_jd",
]

