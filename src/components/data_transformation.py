import os
import sys
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from src.exception.exception import CustomException
from src.logger.logger import logging
from src.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    vectorizer_path: str = os.path.join("artifacts", "vectorizer", "tfidf.pkl")
    label_encoder_path: str = os.path.join("artifacts", "vectorizer", "label_encoder.pkl")


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def clean_text(self, text):
        # Handle empty/null values
        if pd.isna(text) or not isinstance(text, str):
            return ""
        # Remove URLs
        text = re.sub(r'http\S+\s*', ' ', text)
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Data transformation started")
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded")

            # Drop null values
            train_df = train_df.dropna(subset=['Resume_str', 'Category'])
            test_df = test_df.dropna(subset=['Resume_str', 'Category'])
            logging.info("Null values dropped")

            # Clean text
            train_df['cleaned_resume'] = train_df['Resume_str'].apply(self.clean_text)
            test_df['cleaned_resume'] = test_df['Resume_str'].apply(self.clean_text)
            logging.info("Text cleaning completed")

            # Save cleaned data
            os.makedirs("artifacts", exist_ok=True)
            train_df.to_csv("artifacts/train_cleaned.csv", index=False)
            test_df.to_csv("artifacts/test_cleaned.csv", index=False)
            logging.info("Cleaned data saved")

            # Label Encoding - fit on full data to avoid unseen labels
            le = LabelEncoder()
            all_categories = pd.concat([train_df['Category'], test_df['Category']])
            le.fit(all_categories)
            train_df['encoded_category'] = le.transform(train_df['Category'])
            test_df['encoded_category'] = le.transform(test_df['Category'])

            # TF-IDF Vectorization
            tfidf = TfidfVectorizer(
                max_features=1500,
                stop_words='english',
                ngram_range=(1, 2)
            )

            X_train = tfidf.fit_transform(train_df['cleaned_resume'])
            X_test = tfidf.transform(test_df['cleaned_resume'])

            y_train = train_df['encoded_category']
            y_test = test_df['encoded_category']
            logging.info("TF-IDF vectorization completed")

            # Save vectorizer and label encoder
            save_object(self.transformation_config.vectorizer_path, tfidf)
            save_object(self.transformation_config.label_encoder_path, le)
            logging.info("Vectorizer and label encoder saved")

            return (
                X_train, X_test,
                y_train, y_test,
                self.transformation_config.label_encoder_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    X_train, X_test, y_train, y_test, _ = transformation.initiate_data_transformation(
        train_path, test_path)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print("Data transformation successful!")