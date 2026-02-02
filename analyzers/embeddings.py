from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import joblib.Memory if available; sinon simple in-memory cache
try:
    from joblib import Memory  # type: ignore
    _JOBLIB_AVAILABLE = True
except Exception:
    Memory = None  # type: ignore
    _JOBLIB_AVAILABLE = False


# Charger config si présent
try:
    from config.iter1_config import EMBEDDING_CACHE_DIR, EMBEDDING_MODEL, EMBEDDING_CACHE_TTL_DAYS, EMBEDDING_CACHE_VERSION  # type: ignore
except Exception:
    EMBEDDING_CACHE_DIR = ".cache/embeddings"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_CACHE_TTL_DAYS = 30
    EMBEDDING_CACHE_VERSION = 1

os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)


def _hash_key(model_name: str, text: str) -> str:
    norm = (model_name + "::" + text.strip()).encode("utf-8")
    return hashlib.sha256(norm).hexdigest()


def _cache_path_for_key(key: str) -> str:
    return os.path.join(EMBEDDING_CACHE_DIR, f"{key}.npy")


def _meta_path_for_key(key: str) -> str:
    return os.path.join(EMBEDDING_CACHE_DIR, f"{key}.meta.json")


def _is_meta_fresh(meta: dict) -> bool:
    try:
        created = float(meta.get("created_at", 0))
        version = int(meta.get("cache_version", 0))
        if version != EMBEDDING_CACHE_VERSION:
            return False
        age_days = (time.time() - created) / (60 * 60 * 24)
        return age_days <= EMBEDDING_CACHE_TTL_DAYS
    except Exception:
        return False


def get_embeddings(texts: List[str], model_name: Optional[str] = None, use_cache: bool = True) -> Optional[np.ndarray]:
    """Encode une liste de textes en vecteurs numpy. Utilise un cache disque si disponible (joblib optional)

    Retourne None si sentence-transformers absent et aucun fallback.
    """
    if model_name is None:
        model_name = EMBEDDING_MODEL

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        logger.debug("sentence-transformers non disponible: %s", exc)
        return None

    # charge le modèle
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:
        logger.debug("Erreur chargement sentence-transformers model %s: %s", model_name, exc)
        return None

    # prepare result list and use cache per-text
    vecs = []
    for t in texts:
        key = _hash_key(model_name, t)
        cache_path = _cache_path_for_key(key)
        meta_path = _meta_path_for_key(key)
        if use_cache and os.path.exists(cache_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                if _is_meta_fresh(meta):
                    arr = np.load(cache_path)
                    vecs.append(arr)
                    continue
            except Exception:
                logger.debug("Cache corruption ou meta invalide pour %s", key)

        # compute and save
        try:
            emb = model.encode([t], convert_to_numpy=True)[0]
            arr = np.array(emb, dtype=np.float32)
            # write atomically
            tmp = cache_path + ".tmp"
            np.save(tmp, arr)
            os.replace(tmp + ".npy", cache_path) if tmp.endswith('.npy') else os.replace(tmp, cache_path)
            meta = {"created_at": time.time(), "cache_version": EMBEDDING_CACHE_VERSION, "model": model_name}
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
            vecs.append(arr)
        except Exception as exc:
            logger.debug("Erreur encodage/écriture cache embedding: %s", exc)
            return None

    if not vecs:
        return None

    try:
        return np.vstack(vecs)
    except Exception as exc:
        logger.debug("Erreur création matrice embeddings: %s", exc)
        return None


def compute_embedding_similarity(cv_text: str, jd_text: str, model_name: Optional[str] = None, use_cache: bool = True) -> dict:
    """Compute similarity entre deux textes via embeddings.

    Retourne dict {'score': float (0..1), 'method': str}
    En cas d'échec, retourne {'score': 0.0, 'method': 'error'}
    """
    if not cv_text or not jd_text:
        return {"score": 0.0, "method": "invalid_input"}

    if model_name is None:
        model_name = EMBEDDING_MODEL

    cv_emb = get_embeddings([cv_text], model_name=model_name, use_cache=use_cache)
    jd_emb = get_embeddings([jd_text], model_name=model_name, use_cache=use_cache)

    if cv_emb is None or jd_emb is None:
        return {"score": 0.0, "method": "unavailable"}

    try:
        # cosine similarity
        a = cv_emb[0]
        b = jd_emb[0]
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return {"score": 0.0, "method": "zero_norm"}
        sim = float(np.dot(a, b) / denom)
        return {"score": max(0.0, min(1.0, sim)), "method": "sentence-transformers"}
    except Exception as exc:
        logger.debug("Erreur calcul similarity embeddings: %s", exc)
        return {"score": 0.0, "method": "error"}
