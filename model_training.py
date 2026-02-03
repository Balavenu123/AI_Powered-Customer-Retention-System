import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,roc_curve,auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('model_training')

class TRAINING:
    def all_models(X_train,X_test,y_train,y_test):
        try:
            classifiers = {
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Naive Bayes": GaussianNB(),
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(criterion='entropy'),
                "Random Forest": RandomForestClassifier(criterion='entropy', n_estimators=10),
                "AdaBoost": AdaBoostClassifier(estimator=LogisticRegression(), n_estimators=10),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=10),
                "XGBoost": XGBClassifier(eval_metric="logloss"),
                "SVM": SVC(kernel="rbf",probability=True,random_state=42)
            }

            plt.figure(figsize=(8, 6))
            best_auc = 0
            best_model = None
            best_model_name = None

            for name, model in classifiers.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Log metrics
                logger.info(f'==================={name}=======================')
                logger.info(f'Accuracy: {accuracy_score(y_test, y_pred)}')
                logger.info(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
                logger.info(f'Classification Report:\n{classification_report(y_test, y_pred)}')
                # Get probability scores for ROC
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                except:
                    y_prob = model.decision_function(X_test)

                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                logger.info(f'AUC Score ({name}): {roc_auc}')
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.2f})')

                #track best model
                if roc_auc > best_auc:
                    best_auc = roc_auc
                    best_model = model
                    best_model_name = name

            # Random guess lineq
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves - All Models')
            plt.legend(loc='lower right')
            plt.show()

            logger.info("==============================================")
            logger.info(f'Best Model Based on AUC: {best_model_name}')
            logger.info(f'Best AUC Score: {best_auc}')
            logger.info("==============================================")

            # Hyperparameter tuning for best model
            logger.info('Starting Hyperparameter Tuning')

            from sklearn.model_selection import GridSearchCV

            param_grids = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10,100],
                    "penalty": ["l1", "l2","elasticnet","none"],
                    "solver": ["liblinear", "lbfgs"],
                    "max_iter": [100,500,1000,2000]
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5]
                },
                "Decision Tree": {
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5, 10]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                },
                "XGBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                },
                "SVM": {
                    "C": [0.1, 1, 10],
                    "gamma": ["scale", "auto"],
                    "kernel": ["rbf"]
                }
            }

            if best_model_name in param_grids:
                grid = GridSearchCV(
                    estimator=best_model,
                    param_grid=param_grids[best_model_name],
                    scoring="roc_auc",
                    cv=5,
                    n_jobs=-1
                )

                grid.fit(X_train, y_train)

                best_model = grid.best_estimator_

                logger.info(f"Best Parameters for {best_model_name}: {grid.best_params_}")
                logger.info(f"Tuned CV AUC Score: {grid.best_score_}")

            # ===============================
            # Train final model on full training data
            # ===============================
            best_model.fit(X_train, y_train)

            # Save final tuned model
            with open("best_model.pkl", "wb") as f:
                pickle.dump(best_model, f)

            logger.info("Final tuned model saved as best_model.pkl")
            # Save the best model


        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')