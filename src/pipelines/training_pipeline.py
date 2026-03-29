import os
import sys
from src.exception.exception import CustomException
from src.logger.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        self.model_evaluation = ModelEvaluation()

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            # Step 1 - Data Ingestion
            print("\n🔄 Step 1: Data Ingestion...")
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            print("✅ Data Ingestion Complete!")

            # Step 2 - Data Transformation
            print("\n🔄 Step 2: Data Transformation...")
            X_train, X_test, y_train, y_test, label_encoder_path = \
                self.data_transformation.initiate_data_transformation(
                    train_path, test_path)
            print("✅ Data Transformation Complete!")

            # Step 3 - Model Training
            print("\n🔄 Step 3: Model Training...")
            score = self.model_trainer.initiate_model_training(
                X_train, X_test, y_train, y_test)
            print(f"✅ Model Training Complete! Score: {score*100:.2f}%")

            # Step 4 - Model Evaluation
            print("\n🔄 Step 4: Model Evaluation...")
            results = self.model_evaluation.evaluate(
                X_test, y_test, label_encoder_path)
            print("✅ Model Evaluation Complete!")

            # Final Summary
            print("\n===== PIPELINE COMPLETE =====")
            print(f"✅ Accuracy  : {results['accuracy']*100:.2f}%")
            print(f"✅ Precision : {results['precision']*100:.2f}%")
            print(f"✅ Recall    : {results['recall']*100:.2f}%")
            print(f"✅ F1 Score  : {results['f1_score']*100:.2f}%")
            print("✅ Model saved to artifacts/model/model.pkl")
            print("✅ Vectorizer saved to artifacts/vectorizer/tfidf.pkl")

            logging.info("Training pipeline completed successfully")

            return results

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()