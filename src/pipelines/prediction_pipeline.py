import os
import sys
import pdfplumber
import re
import pandas as pd
from src.exception.exception import CustomException
from src.logger.logger import logging
from src.utils.utils import load_object


class PredictionPipeline:
    def __init__(self):
        self.model_path      = os.path.join("artifacts", "model", "model.pkl")
        self.vectorizer_path = os.path.join("artifacts", "vectorizer", "tfidf.pkl")
        self.encoder_path    = os.path.join("artifacts", "vectorizer", "label_encoder.pkl")

    def extract_text_from_pdf(self, pdf_path):
        logging.info(f"Extracting text from PDF: {pdf_path}")
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + " "
            logging.info("PDF text extraction complete")
            return text.strip()

        except Exception as e:
            raise CustomException(e, sys)

    def clean_text(self, text):
        try:
            if not isinstance(text, str) or len(text) == 0:
                return ""
            # Remove URLs
            text = re.sub(r'http\S+\s*', ' ', text)
            # Remove HTML tags
            text = re.sub(r'<.*?>', ' ', text)
            # Remove special characters
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            # Lowercase
            text = text.lower()
            # Remove single characters
            text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        except Exception as e:
            raise CustomException(e, sys)

    def calculate_match_score(self, resume_text, job_description):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            # Calculate cosine similarity
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([resume_text, job_description])
            score = cosine_similarity(vectors[0], vectors[1])[0][0]
            return round(score * 100, 2)

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, pdf_path, job_description=""):
        logging.info("Prediction started")
        try:
            # Load model, vectorizer, encoder
            model    = load_object(self.model_path)
            tfidf    = load_object(self.vectorizer_path)
            encoder  = load_object(self.encoder_path)

            # Extract text from PDF
            raw_text = self.extract_text_from_pdf(pdf_path)

            if len(raw_text) < 50:
                raise CustomException("PDF text too short or empty!", sys)

            # Clean text
            cleaned_text = self.clean_text(raw_text)

            # Transform text
            text_vector = tfidf.transform([cleaned_text])

            # Predict category
            prediction = model.predict(text_vector)
            category = encoder.inverse_transform(prediction)[0]

            # Get confidence score
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(text_vector)
                confidence = round(max(proba[0]) * 100, 2)
            else:
                confidence = 0.0

            # Calculate match score
            match_score = 0.0
            if job_description:
                match_score = self.calculate_match_score(
                    cleaned_text, job_description)

            # Decision
            if match_score > 0:
                if match_score >= 60:
                    decision = "✅ Shortlisted"
                elif match_score >= 40:
                    decision = "⚠️ Maybe"
                else:
                    decision = "❌ Rejected"
            else:
                decision = "✅ Shortlisted" if confidence >= 60 else "⚠️ Maybe"

            result = {
                "category"    : category,
                "confidence"  : confidence,
                "match_score" : match_score,
                "decision"    : decision,
                "raw_text"    : raw_text[:500],
            }

            logging.info(f"Prediction complete: {result}")
            return result

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    import sys

    # Test with a sample PDF
    pdf_path = "data/raw/sample_resume.pdf"

    if not os.path.exists(pdf_path):
        print("⚠️ Please place a sample PDF resume at:")
        print(f"   {pdf_path}")
        sys.exit(1)

    pipeline = PredictionPipeline()

    # Test without job description
    print("\n🔍 Testing without job description...")
    result = pipeline.predict(pdf_path)
    print(f"Category   : {result['category']}")
    print(f"Confidence : {result['confidence']}%")
    print(f"Decision   : {result['decision']}")

    # Test with job description
    print("\n🔍 Testing with job description...")
    job_desc = """
    Looking for Software Engineer with Python, 
    Machine Learning, SQL experience.
    Minimum 2 years experience required.
    """
    result = pipeline.predict(pdf_path, job_desc)
    print(f"Category   : {result['category']}")
    print(f"Confidence : {result['confidence']}%")
    print(f"Match Score: {result['match_score']}%")
    print(f"Decision   : {result['decision']}")
