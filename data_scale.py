import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('data_scale')

class DATA_SCALING:
    def scaling_data(X_train,X_test,y_train,y_test):
        try:
            # Identify numeric col
            numeric_cols = X_train.select_dtypes(exclude='object').columns
            logger.info(f'Numeric columns: {list(numeric_cols)}')
            # Scoring function
            def score_scaled_df(df):
                skew = df.skew().abs().mean()
                kurt = df.kurtosis().abs().mean()
                return skew + kurt
            # Available scaler
            scalers = {
                'standard': StandardScaler(),
                'minmax': MinMaxScaler(),
                'robust': RobustScaler(),
                'maxabs': MaxAbsScaler()
            }
            best_score = np.inf
            best_scaler = None
            best_name = None
            # Try all scalers
            for name, scaler in scalers.items():
                X_train_temp = X_train.copy()
                X_test_temp = X_test.copy()

                X_train_temp[numeric_cols] = scaler.fit_transform(X_train_temp[numeric_cols])
                X_test_temp[numeric_cols] = scaler.transform(X_test_temp[numeric_cols])
                score = score_scaled_df(X_train_temp[numeric_cols])
                logger.info(f'{name.upper()} scaler score = {score:.4f}')

                if score < best_score:
                    best_score = score
                    best_scaler = scaler
                    best_name = name
            # Apply best scaler
            logger.info(f'BEST SCALER SELECTED = {best_name.upper()}')

            X_train[numeric_cols] = best_scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = best_scaler.transform(X_test[numeric_cols])
            with open('scaler.pkl','wb') as f:
                pickle.dump(best_scaler,f)

            logger.info("Feature stats AFTER scaling:\n" +X_train[numeric_cols].describe().to_string())
            logger.info(f"Final X_train Shape: {X_train.shape}")
            logger.info(f"Final X_test Shape: {X_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')