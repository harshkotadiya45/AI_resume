import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.exception.exception import CustomException
from src.logger.logger import logging
from src.utils.utils import load_object


@dataclass
class ModelEvaluationConfig:
    report_path: str = os.path.join("artifacts", "evaluation")


class ModelEvaluation:
    def __init__(self):
        self.evaluation_config = ModelEvaluationConfig()

    def evaluate(self, X_test, y_test, label_encoder_path):
        logging.info("Model evaluation started")
        try:
            # Load model
            model = load_object(os.path.join("artifacts", "model", "model.pkl"))
            le = load_object(label_encoder_path)

            # Predictions
            y_pred = model.predict(X_test)

            # Metrics
            accuracy  = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall    = recall_score(y_test, y_pred, average='weighted')
            f1        = f1_score(y_test, y_pred, average='weighted')

            # Print results
            print("\n===== MODEL EVALUATION RESULTS =====")
            print(f"Accuracy  : {accuracy*100:.2f}%")
            print(f"Precision : {precision*100:.2f}%")
            print(f"Recall    : {recall*100:.2f}%")
            print(f"F1 Score  : {f1*100:.2f}%")

            # Classification Report
            print("\n===== CLASSIFICATION REPORT =====")
            target_names = le.classes_
            print(classification_report(y_test, y_pred, target_names=target_names))

            # Save evaluation report
            os.makedirs(self.evaluation_config.report_path, exist_ok=True)

            # Save metrics to txt file
            report_file = os.path.join(self.evaluation_config.report_path, "metrics.txt")
            with open(report_file, "w") as f:
                f.write("===== MODEL EVALUATION RESULTS =====\n")
                f.write(f"Accuracy  : {accuracy*100:.2f}%\n")
                f.write(f"Precision : {precision*100:.2f}%\n")
                f.write(f"Recall    : {recall*100:.2f}%\n")
                f.write(f"F1 Score  : {f1*100:.2f}%\n")
                f.write("\n===== CLASSIFICATION REPORT =====\n")
                f.write(classification_report(y_test, y_pred, target_names=target_names))

            logging.info(f"Accuracy: {accuracy} Precision: {precision} Recall: {recall} F1: {f1}")

            # Confusion Matrix
            self.plot_confusion_matrix(y_test, y_pred, le.classes_)

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

        except Exception as e:
            raise CustomException(e, sys)

    def plot_confusion_matrix(self, y_test, y_pred, classes):
        try:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(16, 12))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=classes,
                yticklabels=classes
            )
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(self.evaluation_config.report_path, "confusion_matrix.png")
            plt.savefig(plot_path)
            plt.show()
            logging.info(f"Confusion matrix saved at: {plot_path}")
            print(f"\n✅ Confusion matrix saved at: {plot_path}")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    # Data Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Data Transformation
    transformation = DataTransformation()
    X_train, X_test, y_train, y_test, label_encoder_path = transformation.initiate_data_transformation(
        train_path, test_path)

    # Model Evaluation
    evaluator = ModelEvaluation()
    results = evaluator.evaluate(X_test, y_test, label_encoder_path)