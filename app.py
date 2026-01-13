from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename

import os
import logging
import importlib
import re
from typing import List, Optional

app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# Dossier pour sauvegarder temporairement les fichiers uploadés
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


# Remplacer la route GET actuelle par GET+POST pour traiter le formulaire
@app.route('/improve', methods=['GET', 'POST'])
def improve():
    # POST: traiter l'upload et le texte de la JD
    if request.method == 'POST':
        # Récupérer le fichier CV et la job description depuis le formulaire
        cv_file = request.files.get('cv_file')
        jd_text = request.form.get('job_description', '').strip()

        cv_filename = None
        cv_sections = None
        cv_lang = None
        cv_text = ''

        tfidf_res = None
        embed_sim = None
        compare_res = None
        tfidf_method = None
        embed_method = None

        # Importer les parsers (try/except pour tolérer l'absence éventuelle)
        try:
            from parsers.cv_parser import parse_cv, clean_text, detect_sections, detect_language  # type: ignore
        except Exception:
            parse_cv = None
            clean_text = None
            detect_sections = None
            detect_language = None

        try:
            from parsers.jd_parser import parse_jd  # type: ignore
        except Exception:
            parse_jd = None

        # Importer les analyzers (fonctions principales) via importlib pour robustesse
        try:
            mod_sim = importlib.import_module('analyzers.similarity_scorer')
            compare_cv_jd = getattr(mod_sim, 'compare_cv_jd', None)
            compute_tfidf_similarity = getattr(mod_sim, 'compute_tfidf_similarity', None)
            compute_embedding_similarity = getattr(mod_sim, 'compute_embedding_similarity', None)
            logger.debug('analyzers.similarity_scorer loaded: compare=%s, tfidf=%s, embed=%s', bool(compare_cv_jd), bool(compute_tfidf_similarity), bool(compute_embedding_similarity))
        except Exception as exc:
            logger.debug('Unable to import analyzers.similarity_scorer: %s', exc)
            compare_cv_jd = None
            compute_tfidf_similarity = None
            compute_embedding_similarity = None

        try:
            mod_kw = importlib.import_module('analyzers.keywords_extractor')
            extract_keywords_for_cv = getattr(mod_kw, 'extract_keywords_for_cv', None)
            extract_keywords_for_text = getattr(mod_kw, 'extract_keywords_for_text', None)
            logger.debug('analyzers.keywords_extractor loaded: extract_cv=%s, extract_text=%s', bool(extract_keywords_for_cv), bool(extract_keywords_for_text))
        except Exception as exc:
            logger.debug('Unable to import analyzers.keywords_extractor: %s', exc)
            extract_keywords_for_cv = None
            extract_keywords_for_text = None

        # --- FALLBACKS LOCAUX (garantir résultats même sans analyzers externes) ---
        def _tokenize_local(text: str):
            txt = (text or '').lower()
            return [t for t in __import__('re').sub(r"[^\w\s]", ' ', txt).split() if t]

        def _tfidf_fallback(cv_text_local: str, jd_text_local: str, lang: Optional[str] = None, top_n: int = 20):
            try:
                # compute tf
                jd_tokens = _tokenize_local(jd_text_local)
                cv_tokens = _tokenize_local(cv_text_local)
                if not jd_tokens or not cv_tokens:
                    return {"score": 0.0, "jd_top_terms": [], "cv_top_terms": [], "tfidf_method": "inline-fallback"}
                from math import log, sqrt
                # compute idf
                docs = [set(jd_tokens), set(cv_tokens)]
                df = {}
                for s in docs:
                    for t in s:
                        df[t] = df.get(t, 0) + 1
                idf = {t: log((len(docs) + 1) / (df.get(t, 0) + 1)) + 1.0 for t in set(jd_tokens + cv_tokens)}
                # tf
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
                # cosine
                num = sum(a * b for a, b in zip(jd_vec, cv_vec))
                denom = (sqrt(sum(a * a for a in jd_vec)) * sqrt(sum(b * b for b in cv_vec)))
                score = float(num / denom) if denom != 0 else 0.0
                # top terms
                jd_scores = {w: jd_tf.get(w, 0.0) * idf.get(w, 1.0) for w in vocab}
                cv_scores = {w: cv_tf.get(w, 0.0) * idf.get(w, 1.0) for w in vocab}
                jd_top = [k for k, _ in sorted(jd_scores.items(), key=lambda x: x[1], reverse=True)[:top_n] if jd_scores[k] > 0]
                cv_top = [k for k, _ in sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)[:top_n] if cv_scores[k] > 0]
                return {"score": score, "jd_top_terms": jd_top, "cv_top_terms": cv_top, "tfidf_method": "inline-fallback"}
            except Exception as e:
                logger.debug('tfidf_fallback error: %s', e)
                return {"score": 0.0, "jd_top_terms": [], "cv_top_terms": [], "tfidf_method": "inline-fallback"}

        def _embed_fallback(cv_text_local: str, jd_text_local: str):
            try:
                s1 = set(_tokenize_local(jd_text_local))
                s2 = set(_tokenize_local(cv_text_local))
                if not s1 and not s2:
                    return {"score": 0.0, "method": "jaccard"}
                inter = len(s1 & s2)
                union = len(s1 | s2)
                score = inter / union if union != 0 else 0.0
                return {"score": score, "method": "jaccard"}
            except Exception as e:
                logger.debug('embed_fallback error: %s', e)
                return {"score": 0.0, "method": "jaccard"}

        def _keywords_fallback(text_local: str, top_n: int = 12, methods: Optional[List[str]] = None, lang: Optional[str] = None):
            try:
                tokens = _tokenize_local(text_local)
                if not tokens:
                    return {}
                from collections import Counter
                c = Counter(tokens)
                total = sum(c.values())
                items = [(k, v / total) for k, v in c.most_common(top_n)]
                return {'frequency': items}
            except Exception as e:
                logger.debug('keywords_fallback error: %s', e)
                return {}

        # ensure functions available
        if compute_tfidf_similarity is None:
            compute_tfidf_similarity = _tfidf_fallback
            local_tfidf = True
        else:
            local_tfidf = False
        if compute_embedding_similarity is None:
            compute_embedding_similarity = _embed_fallback
            local_embed = True
        else:
            local_embed = False
        if extract_keywords_for_text is None:
            extract_keywords_for_text = _keywords_fallback
            local_kw = True
        else:
            local_kw = False
        if extract_keywords_for_cv is None:
            extract_keywords_for_cv = lambda src, from_file=False, top_n=12: _keywords_fallback(src, top_n=top_n)
            local_kw_cv = True
        else:
            local_kw_cv = False

        # Si pas de fichier, lire le champ de texte collé
        cv_text_input = request.form.get('cv_text_input', '').strip()

        # Sauvegarder le fichier uploadé si présent
        if cv_file and cv_file.filename:
            filename = secure_filename(cv_file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv_file.save(save_path)
            cv_filename = filename

            # Traiter selon l'extension
            ext = os.path.splitext(filename)[1].lower()
            try:
                if parse_cv is not None and ext in ('.pdf', '.docx'):
                    parsed = parse_cv(save_path)
                    cv_text = parsed.get('text', '')
                    cv_lang = parsed.get('lang')
                    cv_sections = parsed.get('sections')
                else:
                    # Pour les .txt/.md/.doc ou si parse_cv manquant, lire comme texte simple
                    if ext in ('.txt', '.md', '.doc') or parse_cv is None:
                        with open(save_path, 'r', encoding='utf-8', errors='replace') as fh:
                            raw = fh.read()
                        if clean_text is not None:
                            cv_text = clean_text(raw)
                        else:
                            cv_text = raw
                        if detect_language is not None:
                            cv_lang = detect_language(cv_text)
                        if detect_sections is not None:
                            cv_sections = detect_sections(cv_text, lang=cv_lang)
            except Exception as exc:
                logger.exception('Erreur pendant l\'analyse du CV: %s', exc)
                cv_text = f"Erreur pendant l'analyse du CV: {exc}"
            # fallback: si parse_cv a échoué et cv_text vide, tenter lecture brute
            if not cv_text:
                try:
                    with open(save_path, 'r', encoding='utf-8', errors='replace') as fh:
                        raw = fh.read()
                    cv_text = clean_text(raw) if clean_text else raw
                except Exception as exc:
                    logger.debug('Impossible de lire brute le fichier upload: %s', exc)

        # Traiter la Job Description
        jd_parsed = {'text': jd_text, 'lang': None}
        if jd_text and 'parse_jd' in locals() and parse_jd is not None:
            try:
                jd_parsed = parse_jd(jd_text, from_file=False)
            except Exception:
                jd_parsed = {'text': jd_text, 'lang': None}

        # If user pasted CV text into the form, prefer it when no file provided
        if not cv_text and cv_text_input:
            if clean_text is not None:
                cv_text = clean_text(cv_text_input)
            else:
                cv_text = cv_text_input

        if not cv_text:
            logger.debug('Aucun texte CV extrait — vérifie si l utilisateur a uploadé un fichier ou collé le CV.')

        # Calculer comparaison (TF-IDF + embeddings + mots-clés) — appel déterministe aux fonctions disponibles
        tfidf = {}
        try:
            if cv_text and jd_parsed.get('text'):
                # TF-IDF
                if compute_tfidf_similarity is not None:
                    try:
                        tfidf = compute_tfidf_similarity(cv_text, jd_parsed.get('text'), lang=cv_lang, top_n=25)
                        # If module implementation returns None, use inline fallback
                        if tfidf is None:
                            logger.debug('compute_tfidf_similarity returned None, using inline fallback')
                            tfidf = _tfidf_fallback(cv_text, jd_parsed.get('text'), lang=cv_lang, top_n=25)
                        if isinstance(tfidf, dict):
                            tfidf_res = tfidf.get('score')
                            tfidf_method = tfidf.get('tfidf_method')
                            compare_res = compare_res or {}
                            compare_res['tfidf_score'] = tfidf.get('score')
                            compare_res.setdefault('jd_top_terms', tfidf.get('jd_top_terms'))
                        # ensure method flagged when we have a score
                        if tfidf_res is not None and not tfidf_method:
                            tfidf_method = 'inline-fallback'
                    except Exception as exc:
                        logger.debug('compute_tfidf_similarity error: %s', exc)

                # Embeddings
                if compute_embedding_similarity is not None:
                    try:
                        embed_res = compute_embedding_similarity(cv_text, jd_parsed.get('text'))
                        # If module returns None or non-dict, fallback to local
                        if embed_res is None:
                            logger.debug('compute_embedding_similarity returned None, using inline fallback')
                            embed_res = _embed_fallback(cv_text, jd_parsed.get('text') )
                        if isinstance(embed_res, dict):
                            embed_sim = embed_res.get('score')
                            embed_method = embed_res.get('method')
                            compare_res = compare_res or {}
                            compare_res['embed_score'] = embed_sim
                            compare_res['embed_method'] = embed_method
                        else:
                            embed_sim = embed_res
                        # ensure method flagged when we have a score
                        if embed_sim is not None and not embed_method:
                            embed_method = 'inline-fallback'
                    except Exception as exc:
                        logger.debug('compute_embedding_similarity error: %s', exc)

                # JD keywords via extractor (preferred) or tfidf fallback
                jd_keywords: List[str] = []
                if extract_keywords_for_text is not None:
                    try:
                        ke_jd = extract_keywords_for_text(jd_parsed.get('text'), top_n=25, methods=None, lang=jd_parsed.get('lang'))
                        merged: List[str] = []
                        for method in ('yake', 'keybert', 'spacy', 'frequency'):
                            for k, _s in (ke_jd.get(method) or []):
                                if k not in merged:
                                    merged.append(k)
                        jd_keywords = merged[:25]
                    except Exception as exc:
                        logger.debug('extract_keywords_for_text error: %s', exc)

                if not jd_keywords and isinstance(tfidf, dict):
                    jd_keywords = tfidf.get('jd_top_terms', [])[:25]

                compare_res = compare_res or {}
                compare_res['jd_top_terms'] = jd_keywords

                # compute present/missing
                present = []
                missing = []
                norm_cv = re.sub(r"[^\w\s]", " ", (cv_text or "")).lower()
                for term in jd_keywords:
                    t = re.sub(r"[^\w\s]", " ", term).lower().strip()
                    if re.search(rf"\b{re.escape(t)}\b", norm_cv):
                        present.append(term)
                    else:
                        missing.append(term)
                compare_res['present_terms'] = present
                compare_res['missing_terms'] = missing
        except Exception as exc:
            logger.exception('Erreur lors du calcul des similarités: %s', exc)
            compare_res = None
            tfidf_res = None
            embed_sim = None
            tfidf_method = None
            embed_method = None

        # keywords CV: prefer extract_keywords_for_cv then extract_keywords_for_text
        cv_keywords = None
        try:
            if extract_keywords_for_cv is not None and cv_text:
                cv_keywords = extract_keywords_for_cv(cv_text, from_file=False, top_n=12)
            elif extract_keywords_for_text is not None and cv_text:
                cv_keywords = extract_keywords_for_text(cv_text, top_n=12, methods=None, lang=cv_lang)
        except Exception as exc:
            logger.debug('Erreur extraction keywords CV: %s', exc)
            cv_keywords = None

        # If cv_keywords is empty dict, try frequency fallback explicitly
        try:
            if (not cv_keywords or (isinstance(cv_keywords, dict) and not any(cv_keywords.values()))) and cv_text:
                freq = _keywords_fallback(cv_text, top_n=12)
                if freq:
                    cv_keywords = freq
        except Exception:
            pass

        # Final safety: if scores still None but functions available, compute them now
        try:
            if tfidf_res is None and compute_tfidf_similarity is not None and cv_text and jd_parsed.get('text'):
                tfidf = compute_tfidf_similarity(cv_text, jd_parsed.get('text'), lang=cv_lang, top_n=25)
                if isinstance(tfidf, dict):
                    tfidf_res = tfidf.get('score')
                    tfidf_method = tfidf.get('tfidf_method')
                    compare_res = compare_res or {}
                    compare_res.setdefault('jd_top_terms', tfidf.get('jd_top_terms'))
            if embed_sim is None and compute_embedding_similarity is not None and cv_text and jd_parsed.get('text'):
                embed_res = compute_embedding_similarity(cv_text, jd_parsed.get('text'))
                if isinstance(embed_res, dict):
                    embed_sim = embed_res.get('score')
                    embed_method = embed_res.get('method')
                    compare_res = compare_res or {}
                    compare_res.setdefault('embed_score', embed_sim)
                    compare_res.setdefault('embed_method', embed_method)
                else:
                    embed_sim = embed_res
        except Exception as exc:
            logger.debug('Final safety compute failed: %s', exc)

        # Défauts finaux pour garantir affichage
        if tfidf_res is None:
            tfidf_res = 0.0
        if embed_sim is None:
            embed_sim = 0.0
        if compare_res is None:
            compare_res = { 'jd_top_terms': [], 'present_terms': [], 'missing_terms': [] }
        if cv_keywords is None:
            cv_keywords = {}
        if tfidf_method is None:
            tfidf_method = 'unavailable'
        if embed_method is None:
            embed_method = 'unavailable'

        # debug: log computed values
        logger.debug('tfidf_res=%s tfidf_method=%s embed_sim=%s embed_method=%s compare_res_keys=%s cv_keywords_keys=%s cv_text_len=%s',
                     tfidf_res, tfidf_method, embed_sim, embed_method, list(compare_res.keys()) if compare_res else None, list(cv_keywords.keys()) if isinstance(cv_keywords, dict) else None, len(cv_text or ''))

        return render_template(
            'improve.html',
            cv_filename=cv_filename,
            job_description=jd_parsed.get('text'),
            cv_sections=cv_sections,
            cv_lang=cv_lang,
            cv_text=cv_text,
            jd_lang=jd_parsed.get('lang'),
            tfidf_score=tfidf_res,
            embed_score=embed_sim,
            compare_res=compare_res,
            tfidf_method=tfidf_method,
            embed_method=embed_method,
            cv_keywords=cv_keywords,
        )

    # GET: afficher la page (vide ou default)
    return render_template('improve.html',
                           cv_filename=None,
                           job_description=None,
                           cv_sections=None,
                           cv_lang=None,
                           cv_text=None,
                           jd_lang=None,
                           tfidf_score=0.0,
                           embed_score=0.0,
                           compare_res={'jd_top_terms': [], 'present_terms': [], 'missing_terms': []},
                           tfidf_method='unavailable',
                           embed_method='unavailable',
                           cv_keywords={})


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'cv' not in request.files or 'job_description' not in request.files:
        return redirect(request.url)
    cv = request.files['cv']
    job_description = request.files['job_description']
    # Process the files here
    return 'Files uploaded successfully'


if __name__ == '__main__':
    # démarrer en mode debug pour voir les logs
    app.run(debug=True)
