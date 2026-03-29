import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from src.exception.exception import CustomException
from src.logger.logger import logging
from src.utils.utils import save_object


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        logging.info("Model training started")
        try:
            models_params = {
                "Logistic Regression": {
                    "model": LogisticRegression(
                        max_iter=1000,
                        random_state=42
                    ),
                    "params": {
                        "C": [0.1, 1, 10, 100],
                        "solver": ["saga", "lbfgs"],
                        "penalty": ["l2"],
                        "tol": [1e-4, 1e-3]
                    }
                },
                "SVM": {
                    "model": SVC(
                        probability=True,
                        random_state=42
                    ),
                    "params": {
                        "C": [0.1, 1, 10, 100],
                        "kernel": ["linear"],
                        "gamma": ["scale", "auto"],
                        "tol": [1e-4, 1e-3],
                        "decision_function_shape": ["ovr", "ovo"]
                    }
                },
                "LinearSVC": {
                    "model": CalibratedClassifierCV(
                        LinearSVC(random_state=42, max_iter=10000)
                    ),
                    "params": {
                        "estimator__C": [0.01, 0.1, 1, 10, 100],
                        "estimator__loss": ["hinge", "squared_hinge"],
                        "estimator__tol": [1e-4, 1e-3],
                        "estimator__max_iter": [1000, 2000]
                    }
                },
                "Multinomial NB": {
                    "model": MultinomialNB(),
                    "params": {
                        "alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
                        "fit_prior": [True, False]
                    }
                }
            }

            # Track best model
            best_score = 0
            best_model = None
            best_model_name = None
            report = {}

            print("\n===== GRID SEARCH CV RESULTS =====")

            for name, mp in models_params.items():
                print(f"\n🔍 Tuning {name}...")

                try:
                    gs = GridSearchCV(
                        mp["model"],
                        mp["params"],
                        cv=3,
                        n_jobs=-1,
                        verbose=0,
                        scoring="accuracy"
                    )
                    gs.fit(X_train, y_train)

                    # Test accuracy
                    y_pred = gs.best_estimator_.predict(X_test)
                    test_score = accuracy_score(y_test, y_pred)

                    report[name] = test_score

                    print(f"   Best Params: {gs.best_params_}")
                    print(f"   CV Score:    {gs.best_score_*100:.2f}%")
                    print(f"   Test Score:  {test_score*100:.2f}%")

                    logging.info(f"{name} - Best Params: {gs.best_params_} - Test Score: {test_score}")

                    # Track best model
                    if test_score > best_score:
                        best_score = test_score
                        best_model = gs.best_estimator_
                        best_model_name = name

                except Exception as model_error:
                    print(f"   ⚠️ {name} failed: {model_error}")
                    logging.warning(f"{name} failed: {model_error}")
                    continue

            # Print final summary
            print("\n===== FINAL SUMMARY =====")
            for name, score in report.items():
                print(f"{name:<25} {score*100:.2f}%")

            print(f"\n✅ Best Model: {best_model_name}")
            print(f"✅ Best Score: {best_score*100:.2f}%")

            # Check minimum accuracy
            if best_score < 0.6:
                raise CustomException("No best model found!", sys)

            # Save best model
            save_object(self.model_trainer_config.model_path, best_model)
            logging.info(f"Best model saved: {best_model_name}")

            return best_score

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
    X_train, X_test, y_train, y_test, _ = transformation.initiate_data_transformation(
        train_path, test_path)

    # Model Training
    trainer = ModelTrainer()
    score = trainer.initiate_model_training(X_train, X_test, y_train, y_test)