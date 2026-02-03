import os
import sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('data_balancing')

class DATA_BALANCE:
    def balancing_data(X_train,y_train):
        try:
            logger.info(f'Before Number of Rows for Good class : {sum(y_train == 1)}')
            logger.info(f'Before Number of Rows for Bad class : {sum(y_train == 0)}')
            sm_res = SMOTE(random_state=42)
            X_train_bal, y_train_bal = sm_res.fit_resample(X_train, y_train)
            logger.info(f'After Number of Rows for Good class : {sum(y_train_bal == 1)}')
            logger.info(f'After Number of Rows for Bad class : {sum(y_train_bal == 0)}')
            logger.info(f'{X_train_bal.shape}')
            logger.info(f'{y_train_bal.shape}')
            logger.info(f'{X_train_bal.sample(10)}')
            return X_train_bal, y_train_bal
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')