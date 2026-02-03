from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Optional diskcache
try:
    import diskcache as dc  # type: ignore
    _DISKCACHE_AVAILABLE = True
except Exception:
    dc = None  # type: ignore
    _DISKCACHE_AVAILABLE = False

# Lazy import joblib.Memory if available (kept for backward compatibility)
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

# Chunking defaults (chars heuristic)
DEFAULT_CHUNK_SIZE = 3000  # approx 200-500 tokens depending on text
DEFAULT_CHUNK_OVERLAP = 200
BATCH_SIZE = 32

os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)


# Internal model singleton
_MODEL = None
_MODEL_NAME = None

# diskcache instance if available
_CACHE = None
if _DISKCACHE_AVAILABLE:
    try:
        _CACHE = dc.Cache(EMBEDDING_CACHE_DIR)
    except Exception:
        _CACHE = None


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


def _chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Découpe un grand texte en chunks basés sur paragraphes pour éviter de dépasser les limites."""
    if not text:
        return []
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # split on double newlines (paragraphs) as priority
    parts = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks: List[str] = []
    current = ""
    for p in parts:
        if not current:
            current = p
            continue
        if len(current) + 2 + len(p) <= max_chars:
            current = current + "\n\n" + p
        else:
            chunks.append(current)
            current = p
    if current:
        chunks.append(current)

    # If still chunks too large (very long paragraph), split by window
    out: List[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            out.append(c)
            continue
        start = 0
        L = len(c)
        while start < L:
            end = min(L, start + max_chars)
            out.append(c[start:end])
            start = end - overlap if end < L else end
    return out


def _load_model(model_name: str):
    """Charge (ou récupère) le modèle d'embeddings en déléguant au `app.get_model()` s'il existe.

    Pour respecter la contrainte de chargement paresseux centralisé, on évite d'importer
    `SentenceTransformer` ici et on délègue à `app.get_model()` (défini dans `app.py`).
    """
    global _MODEL, _MODEL_NAME
    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL

    # Tentative: déléguer à app.get_model() si disponible
    try:
        # import local pour éviter résolution au démarrage
        import app as _app  # type: ignore
        get_m = getattr(_app, 'get_model', None)
        if callable(get_m):
            try:
                m = get_m()
                if m is not None:
                    _MODEL = m
                    _MODEL_NAME = model_name
                    return _MODEL
            except Exception as exc:
                logger.debug("app.get_model() a levé une exception: %s", exc)
    except Exception as exc:
        logger.debug("Impossible d'importer app pour déléguer le chargement du modèle: %s", exc)

    # Si on arrive ici, on ne peut pas charger de modèle
    logger.debug("Aucun modèle embeddings disponible (delegated loader failed)")
    return None


def get_model(model_name: Optional[str] = None):
    """Wrapper public pour récupérer le singleton du modèle.

    `get_model` est la fonction que les autres modules doivent appeler.
    Elle délègue à `_load_model` et applique le nom de modèle par défaut si None.
    Retourne l'instance du modèle ou None si la librairie n'est pas disponible.
    """
    if model_name is None:
        model_name = EMBEDDING_MODEL
    return _load_model(model_name)


def preload_model(model_name: Optional[str] = None, background: bool = False) -> None:
    """Precharge le modèle en mémoire. Si background=True lance dans un thread separé."""
    if model_name is None:
        model_name = EMBEDDING_MODEL

    if background:
        try:
            import threading

            t = threading.Thread(target=_load_model, args=(model_name,), daemon=True)
            t.start()
            logger.debug("Preload model en background: %s", model_name)
            return
        except Exception:
            pass

    _load_model(model_name)


def get_embeddings(texts: List[str], model_name: Optional[str] = None, use_cache: bool = True) -> Optional[np.ndarray]:
    """Encode une liste de textes en vecteurs numpy. Utilise un cache disque si disponible.

    Retourne None si sentence-transformers absent et aucun fallback.
    """
    if model_name is None:
        model_name = EMBEDDING_MODEL

    model = _load_model(model_name)
    if model is None:
        return None

    vecs = []
    for t in texts:
        key = _hash_key(model_name, t)

        # 1) try diskcache
        if use_cache and _CACHE is not None:
            try:
                val = _CACHE.get(key)
                meta = _CACHE.get(key + ":meta")
                if val is not None and meta and _is_meta_fresh(meta):
                    arr = np.array(val, dtype=np.float32)
                    vecs.append(arr)
                    continue
            except Exception:
                logger.debug("diskcache get failed for key %s", key)

        # 2) try file-based cache
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

        # compute embedding (with chunking if necessary)
        try:
            chunks = _chunk_text(t)
            if not chunks:
                arr = np.zeros((model.get_sentence_embedding_dimension(),), dtype=np.float32)
                vecs.append(arr)
                continue

            # encode in batches
            all_embs = []
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i : i + BATCH_SIZE]
                embs = model.encode(batch, convert_to_numpy=True)
                # ensure 2D
                if embs.ndim == 1:
                    embs = np.expand_dims(embs, 0)
                all_embs.append(embs)
            all_embs = np.vstack(all_embs)
            # aggregate: mean pooling across chunks
            arr = np.mean(all_embs, axis=0).astype(np.float32)

            # save to cache
            try:
                meta = {"created_at": time.time(), "cache_version": EMBEDDING_CACHE_VERSION, "model": model_name}
                if _CACHE is not None:
                    try:
                        _CACHE.set(key, arr.tolist())
                        _CACHE.set(key + ":meta", meta)
                    except Exception:
                        logger.debug("diskcache set failed for %s", key)
                else:
                    tmp = cache_path + ".tmp"
                    np.save(tmp, arr)
                    # np.save adds .npy extension automatically
                    if tmp.endswith('.npy'):
                        os.replace(tmp, cache_path)
                    else:
                        os.replace(tmp + '.npy', cache_path)
                    with open(meta_path, "w", encoding="utf-8") as fh:
                        fh.write(json.dumps(meta))
            except Exception:
                logger.debug("Echec enregistrement cache pour %s", key)

            vecs.append(arr)
        except Exception as exc:
            logger.debug("Erreur encodage embedding pour un texte: %s", exc)
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
