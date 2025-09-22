import pdfplumber
import fitz  # PyMuPDF
import spacy
import string
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")


# -----------------------------
# 1. PDF Extraction Functions
# -----------------------------
def extract_text_pdfplumber(file_path):
    """Try extracting text with pdfplumber"""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        print("pdfplumber failed:", e)
    return text.strip()


def extract_text_fitz(file_path):
    """Fallback: use PyMuPDF (better for tables/layout)"""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            blocks = page.get_text("blocks")
            blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # sort by y, then x
            for b in blocks:
                text += b[4] + " "
    except Exception as e:
        print("fitz failed:", e)
    return text.strip()


def extract_text_from_pdf(file_path):
    """Main extractor with fallback"""
    text = extract_text_pdfplumber(file_path)
    if len(text) < 50:  # If too short, fallback to fitz
        text = extract_text_fitz(file_path)
    return text


# -----------------------------
# 2. Preprocessing
# -----------------------------
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.text not in string.punctuation
        and token.text not in stop_words
        and token.is_alpha
    ]
    return " ".join(tokens)


# -----------------------------
# 3. Keyword Extraction (TF-IDF)
# -----------------------------
def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = tfidf_matrix.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    keywords = [(feature_names[i], scores[i]*100) for i in scores.argsort()[-top_n:][::-1]]
    return keywords


# -----------------------------
# 4. Pipeline Function
# -----------------------------
def resume_keyword_extractor(file_path, top_n=10):
    text = extract_text_from_pdf(file_path)
    cleaned_text = preprocess_text(text)
    keywords = extract_keywords(cleaned_text, top_n)
    return keywords