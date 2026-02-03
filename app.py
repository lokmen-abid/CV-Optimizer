from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import os
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# Dossier pour sauvegarder temporairement les fichiers uploadés
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Charger les domaines disponibles (chemin robuste)
def _list_domains():
    domains_dir = os.path.join(app.root_path, 'config', 'domains')
    domains = []
    try:
        for fname in sorted(os.listdir(domains_dir)):
            if fname.endswith('.json'):
                domains.append(os.path.splitext(fname)[0])
    except Exception as exc:
        logger.debug('Impossible de lister les domaines: %s', exc)
    return domains


@app.route('/')
def index():
    domains = _list_domains()
    return render_template('index.html', domains=domains)


# Précharger le modèle embeddings en background si possible (itération 1 option A)
try:
    from analyzers.embeddings import preload_model  # type: ignore
    try:
        preload_model(background=True)
        logger.debug('Preload embeddings launched in background')
    except Exception as exc:
        logger.debug('Preload embeddings failed to start: %s', exc)
except Exception:
    # embeddings optional
    pass


# Route GET+POST pour traiter l'upload du CV et la JD
@app.route('/improve', methods=['GET', 'POST'])
def improve():
    if request.method == 'POST':
        cv_file = request.files.get('cv_file')
        jd_text = request.form.get('job_description', '').strip()
        domain = request.form.get('domain') or None
        cv_lang_choice = request.form.get('cv_lang') or None
        jd_lang_choice = request.form.get('jd_lang') or None

        cv_filename = None
        cv_sections = None
        cv_lang = None
        cv_text = ''

        ats_score = None
        recommendations = []
        match_results = None
        keywords_df = None

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

        # Importer analyzers locaux
        try:
            from analyzers.ats import calculate_ats_score, generate_recommendations  # type: ignore
        except Exception:
            calculate_ats_score = None
            generate_recommendations = None

        try:
            from analyzers.similarity_scorer import compute_tfidf_similarity, keyword_overlap, compute_embedding_similarity  # type: ignore
        except Exception:
            compute_tfidf_similarity = None
            keyword_overlap = None
            compute_embedding_similarity = None

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
                    cv_text = parsed.get('cleaned_text', '') or parsed.get('raw_text', '')
                    cv_lang = parsed.get('language')
                    cv_sections = parsed.get('sections')
                else:
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

        # override languages if user chose explicit
        if cv_lang_choice and cv_lang_choice != 'auto':
            cv_lang = cv_lang_choice
        if jd_lang_choice and jd_lang_choice != 'auto':
            jd_parsed['lang'] = jd_lang_choice

        # If no CV uploaded, abort early (CV is required)
        if not cv_text:
            logger.debug('Aucun fichier CV uploadé — le téléchargement d\'un CV est requis.')
            return render_template('improve.html', cv_filename=None, job_description=jd_parsed.get('text'), cv_sections=None, cv_lang=None, cv_text=None, jd_lang=jd_parsed.get('lang'))

        # Calculer ATS (maintenant retourne un dict détaillé)
        try:
            if calculate_ats_score is not None:
                ats_res = calculate_ats_score(cv_text, domain=domain, lang=cv_lang)
                ats_score = ats_res.get('total', 0)
            else:
                ats_res = {'total': 0, 'breakdown': {}}
                ats_score = 0
        except Exception as exc:
            logger.debug('Erreur calculate_ats_score: %s', exc)
            ats_res = {'total': 0, 'breakdown': {}}
            ats_score = 0

        # Similarity TF-IDF
        try:
            if compute_tfidf_similarity is not None and jd_parsed.get('text'):
                tfidf_res = compute_tfidf_similarity(cv_text, jd_parsed.get('text'), lang=jd_parsed.get('lang') or cv_lang, top_n=20)
                match_results = {
                    'similarity_score': round((tfidf_res.get('score') or 0.0) * 100.0, 1),
                    'tfidf_method': tfidf_res.get('tfidf_method'),
                    'jd_top_terms': tfidf_res.get('jd_top_terms', []),
                    'cv_top_terms': tfidf_res.get('cv_top_terms', [])
                }
            else:
                match_results = {'similarity_score': 0.0, 'tfidf_method': 'unavailable', 'jd_top_terms': [], 'cv_top_terms': []}
        except Exception as exc:
            logger.debug('Erreur compute_tfidf_similarity: %s', exc)
            match_results = {'similarity_score': 0.0, 'tfidf_method': 'error', 'jd_top_terms': [], 'cv_top_terms': []}

        # Keyword overlap
        try:
            if keyword_overlap is not None and jd_parsed.get('text'):
                kw = keyword_overlap(cv_text, jd_parsed.get('text'))
                match_results['keyword_match'] = kw.get('keyword_match')
                match_results['matching_keywords'] = kw.get('matching_keywords')
            else:
                match_results['keyword_match'] = 0.0
                match_results['matching_keywords'] = []
        except Exception as exc:
            logger.debug('Erreur keyword_overlap: %s', exc)
            match_results['keyword_match'] = 0.0
            match_results['matching_keywords'] = []

        # Embedding similarity (itération 1) - optionnel
        try:
            if compute_embedding_similarity is not None and jd_parsed.get('text'):
                emb = compute_embedding_similarity(cv_text, jd_parsed.get('text'))
                if isinstance(emb, dict):
                    match_results['embed_score'] = round((emb.get('score') or 0.0) * 100.0, 1)
                    match_results['embed_method'] = emb.get('method')
                else:
                    match_results['embed_score'] = 0.0
                    match_results['embed_method'] = 'unavailable'
        except Exception as exc:
            logger.debug('Erreur compute_embedding_similarity: %s', exc)
            match_results['embed_score'] = 0.0
            match_results['embed_method'] = 'error'

        # Recommandations (basées sur ats_res et match_results / jd text)
        try:
            if generate_recommendations is not None:
                recommendations = generate_recommendations(cv_text, ats_res, domain=domain, lang=cv_lang, jd_text=jd_parsed.get('text'), match_results=match_results)
            else:
                recommendations = []
        except Exception as exc:
            logger.debug('Erreur generate_recommendations: %s', exc)
            recommendations = []

        # Format des mots-clés: simple liste
        keywords_table = match_results.get('matching_keywords', [])

        return render_template(
            'improve.html',
            cv_filename=cv_filename,
            job_description=jd_parsed.get('text'),
            cv_sections=cv_sections,
            cv_lang=cv_lang,
            cv_text=cv_text,
            jd_lang=jd_parsed.get('lang'),
            ats_score=ats_score,
            ats_res=ats_res,
            recommendations=recommendations,
            match_results=match_results,
            keywords_table=keywords_table,
        )

    # GET: afficher la page (vide ou default)
    return render_template('improve.html',
                           cv_filename=None,
                           job_description=None,
                           cv_sections=None,
                           cv_lang=None,
                           cv_text=None,
                           jd_lang=None)


# route /upload supprimée (non utilisée) - code nettoyé

if __name__ == '__main__':
    # démarrer en mode debug pour voir les logs
    port = int(os.environ.get('PORT', 5000))  # Render fournit PORT
    app.run(host='0.0.0.0', port=port, debug=False)
