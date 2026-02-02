from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from jinja2 import Template  # type: ignore
except Exception:
    Template = None  # type: ignore


DEFAULT_TEMPLATES = [
    ("add_action_verbs", "Augmente l'impact de tes descriptions en utilisant plus de verbes d'action : {{ verbs }}."),
    ("quantify_results", "Considère d'ajouter des résultats quantifiables : par exemple 'Augmentation de X% / Réduction de Y%' ou chiffres précis."),
    ("highlight_skills", "Mets en avant ces compétences clés demandées: {{ keywords }}."),
    ("formatting", "Utilisez des bullet points pour les réalisations et préférez des phrases courtes et actives."),
    ("contact_info", "Vérifiez que vos informations de contact (email, téléphone, LinkedIn) sont visibles en haut du CV."),
    ("tailor_for_job", "Adaptez votre CV en incorporant des mots-clés présents dans la description du poste: {{ keywords }}."),
]


def generate_recommendations_with_templates(cv_text: str, ats_score: Optional[int] = None, jd_keywords: Optional[List[str]] = None, lang: Optional[str] = None, max_results: int = 5) -> List[str]:
    """Génère des recommandations en appliquant des templates simples.

    Si Jinja2 absent, utilise des templates basiques format().

    Args:
        cv_text: texte du CV (utilisé pour heuristiques simples)
        ats_score: score ATS si disponible
        jd_keywords: mots-clés extraits de la JD
        lang: 'fr' ou 'en' (si None on devine)
        max_results: nombre max de recommandations
    """
    if not cv_text:
        return []

    # heuristique simple de langue si non fournie
    txt = (cv_text or '').lower()
    if lang is None:
        lang = 'fr' if any(w in txt for w in ['profil', 'compétences', 'expérience', 'formation', 'langues', 'résumé']) else 'en'

    # verbs bilingues étendus
    verbs = [
        # EN
        'developed','implemented','managed','led','optimized','designed','created','improved','launched','spearheaded',
        'architected','orchestrated','streamlined','automated','mentored','coordinated','negotiated','resolved','secured','scaled',
        # FR
        'développé','implémenté','géré','dirigé','optimisé','conçu','créé','amélioré','lancé','piloté',
        'architecturé','orchestré','rationalisé','automatisé','encadré','coordonné','négocié','résolu','sécurisé','scalé'
    ]

    # pour l'affichage on prendra une sélection unique
    top_verbs = ", ".join(verbs[:12])
    keywords = ", ".join(jd_keywords or [])

    out: List[str] = []
    for name, tpl in DEFAULT_TEMPLATES:
        try:
            # si la template contient des éléments traduits (FR) on renvoie tel quel; sinon on formate
            if Template is not None:
                rendered = Template(tpl).render(verbs=top_verbs, keywords=keywords, ats_score=ats_score, lang=lang)
            else:
                rendered = tpl.replace("{{ verbs }}", top_verbs).replace("{{ keywords }}", keywords).replace("{{ ats_score }}", str(ats_score or ""))
            out.append(rendered)
        except Exception as exc:
            logger.debug("Erreur génération template %s: %s", name, exc)

    # Ajouts contextuels basés sur le score et la langue
    try:
        word_count = len([w for w in txt.split() if w.strip()])
        if word_count < 400:
            out.append("Votre CV semble court — envisagez d'ajouter des réalisations chiffrées." if lang == 'fr' else "Your CV looks short — consider adding quantified achievements.")
        elif word_count > 1200:
            out.append("Votre CV semble long — concentrez-vous sur les réalisations principales." if lang == 'fr' else "Your CV looks long — focus on main achievements.")

        if ats_score is not None and ats_score < 60:
            out.append("Pensez à incorporer davantage de mots-clés spécifiques au poste et à structurer vos sections." if lang == 'fr' else "Consider incorporating more job-specific keywords and structuring sections clearly.")
    except Exception:
        pass

    # unique and trim to max_results
    seen = set()
    final: List[str] = []
    for s in out:
        if not s or s in seen:
            continue
        seen.add(s)
        final.append(s)
        if len(final) >= max_results:
            break

    return final
