import pytest
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so tests can import local packages like analyzers, parsers
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_DATA_DIR.mkdir(exist_ok=True)

@pytest.fixture
def sample_text_en():
    return (
        "Profile\n"
        "Experienced software engineer with 5+ years in backend development.\n\n"
        "Experience:\n"
        "- Developed APIs in Python and Java.\n"
        "- Implemented CI/CD pipelines and dockerized applications.\n\n"
        "Education:\n"
        "BSc in Computer Science.\n"
    )

@pytest.fixture
def sample_text_fr():
    return (
        "Profil\n"
        "Ingénieur mécanique avec 10 ans d'expérience.\n\n"
        "Expérience:\n"
        "- Conçu et optimisé des systèmes mécaniques.\n"
        "- Réduit les coûts de production de 15%.\n\n"
        "Formation:\n"
        "Diplôme d'ingénieur.\n"
    )

@pytest.fixture
def sample_docx_path(tmp_path, sample_text_en):
    # Crée un fichier .docx de test si python-docx est disponible
    try:
        from docx import Document
    except Exception:
        pytest.skip("python-docx non disponible pour tests DOCX")

    p = tmp_path / "test_resume.docx"
    doc = Document()
    for line in sample_text_en.splitlines():
        doc.add_paragraph(line)
    doc.save(str(p))
    return str(p)

@pytest.fixture
def sample_txt_path(tmp_path, sample_text_en):
    p = tmp_path / "test_resume.txt"
    p.write_text(sample_text_en, encoding="utf-8")
    return str(p)

@pytest.fixture
def sample_pdf_path(tmp_path, sample_text_en):
    # Attempt to create a minimal PDF using reportlab if available, else skip
    try:
        from reportlab.pdfgen import canvas
    except Exception:
        pytest.skip("reportlab non disponible pour créer PDF de test")

    p = tmp_path / "test_resume.pdf"
    c = canvas.Canvas(str(p))
    y = 800
    for line in sample_text_en.splitlines():
        c.drawString(72, y, line)
        y -= 12
    c.save()
    return str(p)
