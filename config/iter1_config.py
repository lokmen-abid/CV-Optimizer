# Configuration optionnelle pour l'Itération 1
# Activez/désactivez les fonctionnalités avancées (keywords, embeddings, spaCy full)
ENABLE_KEYWORD_EXTRACTION = True
ENABLE_SPACY_FULL = True
ENABLE_EMBEDDINGS = True

# Embedding model par défaut (sentence-transformers)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Cache pour embeddings: utilise joblib.Memory (voir requirements)
EMBEDDING_CACHE_DIR = ".cache/embeddings"
EMBEDDING_CACHE_TTL_DAYS = 30

# Priorité des extracteurs de mots-clés
KEYWORD_METHOD_PRIORITY = ["keybert", "yake", "tfidf"]

# Paramètres divers
KEYWORD_TOP_N = 20

# Version du cache (incrémentez si le format change)
EMBEDDING_CACHE_VERSION = 1
