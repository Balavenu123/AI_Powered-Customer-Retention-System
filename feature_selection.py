import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('feature_selection')

class FEATURE:
    def feature_selecting(training_data, testing_data,y_train):
        try:
            logger.info("Feature Selection Started")

            logger.info(f"Train Shape (Before FS): {training_data.shape}")
            logger.info(f"Test Shape (Before FS): {testing_data.shape}")

            # 1. Constant Features
            reg_constant=VarianceThreshold(threshold=0)
            reg_constant.fit(training_data)
            constant_cols = training_data.columns[~reg_constant.get_support()]
            logger.info(f'Constant columns removed: {list(constant_cols)}')

            X_train = pd.DataFrame(reg_constant.transform(training_data),
                                   columns=training_data.columns[reg_constant.get_support()])

            X_test = pd.DataFrame(
                reg_constant.transform(testing_data),
                columns=testing_data.columns[reg_constant.get_support()]
            )
            # logger.info(X_train.isnull().sum())
            # logger.info(X_test.isnull().sum())

            # 2. Quasi-Constant
            reg_quasi_constant=VarianceThreshold(threshold=0.1)
            reg_quasi_constant.fit(X_train)
            quasi_cols = X_train.columns[~reg_quasi_constant.get_support()]
            logger.info(f'Quasi-constant columns removed: {list(quasi_cols)}')

            X_train = pd.DataFrame(reg_quasi_constant.transform(X_train),columns=X_train.columns[reg_quasi_constant.get_support()])
            X_test = pd.DataFrame(reg_quasi_constant.transform(X_test),columns=X_test.columns[reg_quasi_constant.get_support()])

            # 3. Chi-Square Test
            logger.info("Running Chi-Square test")
            if (X_train < 0).any().any():
                logger.warning("Negative values detected. Skipping Chi-Square.")
            else:
                chi_scores, chi_pvalues = chi2(X_train, y_train_numeric)
                chi_pvalues = pd.Series(chi_pvalues, index=X_train.columns)
                chi_alpha = 0.05
                chi_remove = chi_pvalues[chi_pvalues > chi_alpha].index.tolist()
                logger.info(f'Removed by Chi-Square: {chi_remove}')
                X_train = X_train.drop(columns=chi_remove)
                X_test = X_test.drop(columns=chi_remove)

            # 3. Hypothesis Testing
            logger.info(f"Before hypothesis testing: {X_train.shape}")

            y_train_numeric = np.asarray(y_train).ravel()

            p_values = []
            for col in X_train.columns:
                corr, p_val = pearsonr(X_train[col].values, y_train_numeric)
                p_values.append(p_val)
                logger.info(f"{col} | p-value = {p_val:.6f}")

            p_values = pd.Series(p_values, index=X_train.columns)

            alpha = 0.05
            features_to_remove = p_values[p_values > alpha].index.tolist()
            logger.info(f"Removed by hypothesis testing: {features_to_remove}")

            X_train = X_train.drop(columns=features_to_remove)
            X_test = X_test.drop(columns=features_to_remove)

            # Final logs
            logger.info("===================================")
            logger.info("Feature Selection Completed")
            logger.info("===================================")
            logger.info(f"Train Shape (After FS): {X_train.shape}")
            logger.info(f"Test Shape (After FS): {X_test.shape}")
            logger.info(f"Final Columns:\n{X_train.columns}")

            return X_train, X_test
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg} {error_type}')