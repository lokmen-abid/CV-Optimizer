from __future__ import annotations

import re
import logging
import json
from typing import List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)

# NOTE:
# Nous ne stockons plus une liste globale `_ACTION_VERBS`.
# Les verbes proviennent désormais des fichiers JSON de `config/domains/`.

# Section headers heuristiques
_SECTION_HEADERS = ["profile", "profile summary", "summary", "profil", "résumé", "resume", "objective" ,
                    "experience", "work experience", "expérience", "expériences", "professional experience", "employment history" ,
                    "education", "formation", "éducation & certifications", "academic qualifications", "studies", "éducation", "degrees" ,
                    "skills", "compétences", "technical skills", "skills & tools", "compétence", "abilities" , "compétences techniques" ,
                    "languages", "langues", "language skills" ,
                    "projects", "academic projects", "projets", "personal projects", "portfolio" , "projets académiques" ,
                    "contact", "contact", "informations personnelles", "personal info", "contact information" ,
                    "strengths & qualities", "strengths", "qualities", "personal qualities", "atouts" , "qualités professionnelles" ]

_CONTACT_KW = ['phone', 'email', 'linkedin', 'github', 'téléphone', 'courriel']


def _domains_dir() -> Path:
    return Path(__file__).resolve().parents[1] / 'config' / 'domains'


def _load_domain_config(domain: Optional[str]) -> dict:
    """Charge la configuration JSON pour un domaine donné (ex: 'it', 'hr').

    Retourne un dict vide si non trouvé.
    """
    if not domain:
        return {}
    try:
        base = _domains_dir()
        f = base / f"{domain}.json"
        if not f.exists():
            alt = base / f"{domain.lower().replace(' ', '_')}.json"
            if alt.exists():
                f = alt
            else:
                return {}
        with open(f, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception as exc:
        logger.debug("Impossible de charger la config domaine %s: %s", domain, exc)
        return {}


def _collect_all_domain_verbs(lang: Optional[str] = None) -> Set[str]:
    """Agréger les verbes (pour la langue donnée si possible) depuis tous les fichiers JSON de `config/domains/`.

    Retourne un set de verbes en minuscules.
    """
    verbs: Set[str] = set()
    try:
        base = _domains_dir()
        if not base.exists():
            return verbs
        for f in base.iterdir():
            if f.suffix != '.json':
                continue
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                dverbs = data.get('verbs', {})
                if lang and dverbs.get(lang):
                    verbs.update(v.lower() for v in dverbs.get(lang, []))
                else:
                    # ajouter toutes langues disponibles
                    for lst in dverbs.values():
                        verbs.update(v.lower() for v in lst)
            except Exception:
                continue
    except Exception as exc:
        logger.debug("Erreur collecte verbes domaines: %s", exc)
    return verbs


def _get_verbs_for_domain(domain_conf: dict, lang: Optional[str]) -> Set[str]:
    """Retourne l'ensemble des verbes à utiliser: priorise les verbes du domaine (pour la langue si fournie),
    sinon agrège toutes les verbes disponibles dans les domain JSON.
    """
    verbs: Set[str] = set()
    try:
        if domain_conf:
            dverbs = domain_conf.get('verbs', {})
            if lang and dverbs.get(lang):
                verbs.update(v.lower() for v in dverbs.get(lang, []))
            else:
                for lst in dverbs.values():
                    verbs.update(v.lower() for v in lst)
    except Exception:
        pass

    # Si aucun verbe trouvé dans le domaine, agréger tous les domaines
    if not verbs:
        verbs = _collect_all_domain_verbs(lang=lang)

    return verbs


def calculate_ats_score(text: str, domain: Optional[str] = None, lang: Optional[str] = None) -> dict:
    """Calcule un score ATS détaillé et renvoie un dict contenant le score total et le breakdown par section.

    Retourne:
      {
        'total': int,  # note 0-100
        'breakdown': {
           'action_verbs': {'score': int, 'weight': 30, 'pct': float},
           'quantifiable': {...},
           'sections': {...},
           'length': {...},
           'contact': {...}
        }
      }
    """
    if not text:
        return {'total': 0, 'breakdown': {}}
    try:
        txt = (text or '').lower()

        words = re.findall(r"\w+", txt)
        word_count = len(words)

        domain_conf = _load_domain_config(domain)
        verbs = _get_verbs_for_domain(domain_conf, lang)

        # Weights
        weights = {'action_verbs': 30, 'quantifiable': 25, 'sections': 20, 'length': 15, 'contact': 10}

        # action verbs: compter les occurrences de verbes du domaine
        action_verbs_count = sum(1 for w in verbs if w and w in txt)
        if action_verbs_count > 8:
            action_score = weights['action_verbs']
        elif action_verbs_count > 4:
            action_score = int(weights['action_verbs'] * 0.7)
        elif action_verbs_count > 1:
            action_score = int(weights['action_verbs'] * 0.4)
        else:
            action_score = 0

        # quantifiable patterns
        quant_patterns = [r'increased by \d+%', r'reduced by \d+%', r'\d+% improvement', r'saved \$?\d+', r'\d+\+', r'over \d+', r'by \d+', r"\d+\s?%", r"\baugmente?\b", r"\b%\b"]
        quant_count = sum(1 for p in quant_patterns if re.search(p, txt))
        if quant_count > 3:
            quant_score = weights['quantifiable']
        elif quant_count > 1:
            quant_score = int(weights['quantifiable'] * 0.5)
        else:
            quant_score = 0

        # sections presence
        section_count = sum(1 for s in _SECTION_HEADERS if re.search(rf"\b{s}\b", txt))
        if section_count >= 5:
            section_score = weights['sections']
        elif section_count >= 3:
            section_score = int(weights['sections'] * 0.5)
        else:
            section_score = 0

        # length
        if 500 <= word_count <= 800:
            length_score = weights['length']
        elif 400 <= word_count <= 1000:
            length_score = int(weights['length'] * 0.4)
        else:
            length_score = 0

        # contact
        contact_count = sum(1 for k in _CONTACT_KW if re.search(rf"\b{k}\b", txt))
        contact_score = weights['contact'] if contact_count >= 2 else (int(weights['contact'] * 0.5) if contact_count == 1 else 0)

        total = action_score + quant_score + section_score + length_score + contact_score
        total = int(min(total, 100))

        breakdown = {
            'action_verbs': {'score': action_score, 'weight': weights['action_verbs'], 'pct': round(action_score / weights['action_verbs'] * 100 if weights['action_verbs'] else 0, 1), 'count': action_verbs_count},
            'quantifiable': {'score': quant_score, 'weight': weights['quantifiable'], 'pct': round(quant_score / weights['quantifiable'] * 100 if weights['quantifiable'] else 0, 1), 'count': quant_count},
            'sections': {'score': section_score, 'weight': weights['sections'], 'pct': round(section_score / weights['sections'] * 100 if weights['sections'] else 0, 1), 'count': section_count},
            'length': {'score': length_score, 'weight': weights['length'], 'pct': round(length_score / weights['length'] * 100 if weights['length'] else 0, 1), 'word_count': word_count},
            'contact': {'score': contact_score, 'weight': weights['contact'], 'pct': round(contact_score / weights['contact'] * 100 if weights['contact'] else 0, 1), 'count': contact_count},
        }

        return {'total': total, 'breakdown': breakdown}
    except Exception as exc:
        logger.exception("Erreur compute ATS: %s", exc)
        return {'total': 0, 'breakdown': {}}


def generate_recommendations(text: str, ats_res: dict, domain: Optional[str] = None, lang: Optional[str] = None, jd_text: Optional[str] = None, match_results: Optional[dict] = None) -> List[str]:
    """Génère recommandations pertinentes en comparant le CV et la Job Description.

    Cette version sélectionne au maximum 5 recommandations réellement pertinentes :
      - Priorité 0 : mots-clés JD absents du CV
      - Priorité 1 : recommandations spécifiques domaine (si JD le requiert)
      - Priorité 2 : verbes d'action faibles
      - Priorité 3 : peu d'éléments quantifiables
      - Priorité 4 : sections manquantes / mauvaise structure
      - Priorité 5 : coordonnées manquantes
      - Priorité 6 : longueur inadaptée

    L'approche : collecter des items (priorité, message) puis trier et renvoyer les 5 premiers.
    """
    recs: List[str] = []
    cv_text = (text or '').lower()
    jd_text = (jd_text or '').lower() if jd_text else ''

    # heuristique langue si non fournie
    french_indicators = ['profil', 'compétences', 'expérience', 'formation', 'langues', 'résumé', 'objéctif']
    if not lang:
        lang = 'fr' if any(w in cv_text for w in french_indicators) or any(w in jd_text for w in french_indicators) else 'en'

    domain_conf = _load_domain_config(domain)
    # domain recommendations (phrases generales)
    domain_recs: List[str] = []
    try:
        if domain_conf:
            domain_recs = domain_conf.get('recommendations', {}).get(lang, [])
    except Exception:
        domain_recs = []

    # obtain keywords from match_results or compute tfidf if needed
    jd_top = []
    cv_top = []
    try:
        if match_results and isinstance(match_results, dict):
            jd_top = match_results.get('jd_top_terms') or []
            cv_top = match_results.get('cv_top_terms') or []
        else:
            try:
                from analyzers.similarity_scorer import compute_tfidf_similarity
                tf = compute_tfidf_similarity(text, jd_text, lang=lang)
                if tf:
                    jd_top = tf.get('jd_top_terms') or []
                    cv_top = tf.get('cv_top_terms') or []
            except Exception:
                jd_top = []
                cv_top = []
    except Exception:
        jd_top = []
        cv_top = []

    # token sets
    cv_tokens = set(re.findall(r"\w+", cv_text))
    jd_tokens = set(re.findall(r"\w+", jd_text))

    # find missing JD keywords in CV (priority 0)
    missing_jd_terms = [t for t in jd_top if t and t not in cv_tokens]

    # domain verbs
    verbs = _get_verbs_for_domain(domain_conf, lang)
    domain_verbs_in_jd = any(v for v in verbs if v and v in jd_text)
    domain_verbs_missing_in_cv = any(v for v in verbs if v and v in jd_text and v not in cv_text)

    items: List[tuple] = []  # (priority:int, message:str)

    # Priority 0: missing JD keywords
    if missing_jd_terms:
        msg = ("Ajoutez ces mots-clés présents dans la description du poste mais absents du CV : " + ", ".join(missing_jd_terms[:10])) if lang == 'fr' else ("Add these keywords from the job description that are missing in your CV: " + ", ".join(missing_jd_terms[:10]))
        items.append((0, msg))

    # Priority 1: domain-specific recommendations (only if JD contains domain indicators and CV lacks domain verbs/keywords)
    if domain_conf and domain_verbs_in_jd and domain_verbs_missing_in_cv and domain_recs:
        # include up to two domain-specific recommendations
        for dr in (domain_recs[:2]):
            items.append((1, dr))

    # Priority 2: action verbs low
    av_pct = ats_res.get('breakdown', {}).get('action_verbs', {}).get('pct', 0)
    if av_pct < 40:
        msg = "Renforcez l'usage de verbes d'action adaptés à votre domaine pour mieux mettre en valeur vos réalisations." if lang == 'fr' else "Increase use of action verbs relevant to the role to better highlight achievements."
        items.append((2, msg))

    # Priority 3: quantifiable
    q_pct = ats_res.get('breakdown', {}).get('quantifiable', {}).get('pct', 0)
    if q_pct < 40:
        msg = "Ajoutez des résultats quantifiables (%, montants, taux) pour illustrer l'impact de vos actions." if lang == 'fr' else "Add quantifiable results (%, amounts, rates) to illustrate the impact of your work."
        items.append((3, msg))

    # Priority 4: sections
    sec_pct = ats_res.get('breakdown', {}).get('sections', {}).get('pct', 0)
    if sec_pct < 50:
        msg = "Clarifiez et nommez les sections principales (Expérience, Formation, Compétences) pour faciliter l'analyse." if lang == 'fr' else "Clarify and name main sections (Experience, Education, Skills) to ease parsing."
        items.append((4, msg))

    # Priority 5: contact
    contact_pct = ats_res.get('breakdown', {}).get('contact', {}).get('pct', 0)
    if contact_pct < 50:
        msg = "Assurez-vous que vos coordonnées sont clairement visibles (email, téléphone, LinkedIn)." if lang == 'fr' else "Ensure contact details are clearly visible (email, phone, LinkedIn)."
        items.append((5, msg))

    # Priority 6: length
    length_pct = ats_res.get('breakdown', {}).get('length', {}).get('pct', 0)
    if length_pct < 50:
        msg = "Ajustez la longueur du CV : soyez concis tout en gardant des exemples chiffrés." if lang == 'fr' else "Adjust CV length: be concise while keeping quantified examples."
        items.append((6, msg))

    # If no JD provided, rely on the weakest breakdowns only (already captured above), otherwise ensure we prioritize JD-related items
    # Sort items by priority then de-duplicate and limit to 5
    items_sorted = sorted(items, key=lambda x: x[0])
    seen = set()
    final: List[str] = []
    for _, msg in items_sorted:
        if not msg or msg in seen:
            continue
        seen.add(msg)
        final.append(msg)
        if len(final) >= 5:
            break

    # If still empty, add a single gentle generic suggestion
    if not final:
        final_msg = "Votre CV semble correct pour l'ATS, mais pensez à le personnaliser pour chaque offre." if lang == 'fr' else "Your CV seems ATS-friendly; consider tailoring it for each job application."
        final = [final_msg]

    return final
