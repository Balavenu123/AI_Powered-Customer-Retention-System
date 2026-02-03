import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('categorical_to_numerical')

class CAT_TO_NUMERICAL:
    def categori_to_numeric(X_train,X_test):
        try:
            X_train=X_train.copy()
            X_test=X_test.copy()

            col_for_label=['gender','Partner','Dependents','PhoneService','PaperlessBilling']
            label=LabelEncoder()
            for col in col_for_label:
                label.fit(X_train[col])
                X_train[col]=label.transform(X_train[col])
                X_test[col]=label.transform(X_test[col])

            onehot_cols=['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
            logger.info(f'Columns before one-hot: {onehot_cols}')
            for col in onehot_cols:
                logger.info(f'Encoding column: {col}')
                logger.info(f'Unique values in {col}: {X_train[col].unique()}')
                one_hot=OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)
                one_hot.fit(X_train[[col]])
                # -------- TRAIN --------
                train_encoded = one_hot.transform(X_train[[col]])
                train_df=pd.DataFrame(train_encoded,columns=one_hot.get_feature_names_out([col]))
                X_train.reset_index(drop=True, inplace=True)
                train_df.reset_index(drop=True, inplace=True)
                X_train = pd.concat([X_train, train_df], axis=1)
                X_train.drop(columns=[col], inplace=True)
                logger.info(f'X_train columns after encoding {col}: {X_train.columns}')
                # -------- TEST --------
                test_encoded = one_hot.transform(X_test[[col]])
                test_df = pd.DataFrame(test_encoded,columns=one_hot.get_feature_names_out([col]))
                X_test.reset_index(drop=True, inplace=True)
                test_df.reset_index(drop=True, inplace=True)
                X_test = pd.concat([X_test, test_df], axis=1)
                X_test.drop(columns=[col], inplace=True)
                logger.info(f'X_test columns after encoding {col}: {X_test.columns}')

            col_for_ordinal= ['Contract']
            ordinal=OrdinalEncoder(categories=[['Month-to-month','One year','Two year']])
            ordinal.fit(X_train[col_for_ordinal])
            X_train[col_for_ordinal]=ordinal.transform(X_train[col_for_ordinal])
            X_test[col_for_ordinal]=ordinal.transform(X_test[col_for_ordinal])

            col_for_fre=['PaymentMethod','Sim','Region']
            freq_maps={}
            for col in col_for_fre:
                freq_map=X_train[col].value_counts(normalize=True)
                freq_maps[col]=freq_map.to_dict()
                X_train[col]=X_train[col].map(freq_map)
                X_test[col]=X_test[col].map(freq_map).fillna(0)

            with open('freq_maps.pkl','wb') as f:
                pickle.dump(freq_maps,f)

            return X_train,X_test

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')
