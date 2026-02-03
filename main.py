import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('main')
from missing_values import MissingValues
from variable_transformation import VARIABLE_TRANSFORM
from outlier_handling import OUTLIER
from categorical_to_numerical import CAT_TO_NUMERICAL
from feature_selection import FEATURE
from data_balancing import DATA_BALANCE
from data_scale import DATA_SCALING
from model_training import TRAINING

class CustomerRetention:
    def __init__(self,path):
        try:
            self.path=path
            self.df=pd.read_csv(self.path)
            logger.info(f'data loaded {self.df}')
            self.df=self.df.drop(['customerID'],axis=1)
            self.df['Churn'] =self.df['Churn'].map({'No': 0, 'Yes': 1})
            logger.info(f'data loaded {self.df}')
            logger.info(f'shape of data :{self.df.shape}')
            logger.info(f'null values in the data set :{self.df.isnull().sum()}')
            for i in self.df.columns:
                logger.info(f'{self.df[i].dtype}')
            self.y=self.df.iloc[:,-3]
            self.X=self.df.drop('Churn',axis=1)
            logger.info(f'X shape:{self.X.shape}')
            logger.info(f'y shape:{self.y.shape}')
            logger.info(f'X :{self.X}')
            logger.info(f'y :{self.y}')
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            logger.info(f'X_train shape:{self.X_train.shape}')
            logger.info(f'X_test shape:{self.X_test.shape}')
            logger.info(f'y_train shape:{self.y_train.shape}')
            logger.info(f'y_test shape:{self.y_test.shape}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def missingvalues(self):
        try:
            logger.info(f'X_train shape:{self.X_train.shape}')
            logger.info(f'X_test shape:{self.X_test.shape}')
            #logger.info(f'missingvalues :{self.X_train.isnull().sum().any()}')
            if self.X_train.isnull().sum().any() > 0 or self.X_test.isnull().sum().any() > 0:
                self.X_train,self.X_test=MissingValues.null_value_handling(self.X_train,self.X_test)
            else:
                logger.info(f'There are no missing values in X_train and X_test')
            # logger.info(f'X_train null:{self.X_train.isnull().sum()}')
            # logger.info(f'X_test null:{self.X_test.isnull().sum()}')
            logger.info(f'X_train shape:{self.X_train.shape}')
            logger.info(f'X_test shape:{self.X_test.shape}')
            logger.info(f'\n{self.X_train}')
            logger.info(f'\n{self.X_test}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def variable_transform(self):
        try:
            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')
            logger.info(f'\n X_train_num shape:{self.X_train_num.shape}')
            logger.info(f'\nX_test_num shape:{self.X_test_num.shape}')
            logger.info(f'\n X_train_num null values:{self.X_train_num.isnull().sum()}')
            logger.info(f'X_test_num null values:{self.X_test_num.isnull().sum()}')
            self.X_train_num,self.X_test_num=VARIABLE_TRANSFORM.variable_trans(self.X_train_num,self.X_test_num)
            logger.info(f'\n X_train_num shape:{self.X_train_num.shape}')
            logger.info(f'\nX_test_num shape:{self.X_test_num.shape}')
            logger.info(f'\n X_train_num null values:{self.X_train_num.isnull().sum()}')
            logger.info(f'X_test_num null values:{self.X_test_num.isnull().sum()}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def outliers(self):
        try:
            logger.info(f'X_train_num shape:{self.X_train_num.shape}')
            logger.info(f'X_test_num shape:{self.X_test_num.shape}')
            logger.info(f'X_train_num null values:{self.X_train_num.isnull().sum()}')
            logger.info(f'X_test_num null values:{self.X_test_num.isnull().sum()}')
            self.X_train_num,self.X_test_num=OUTLIER.outlier_handle(self.X_train_num,self.X_test_num)
            logger.info(f'\n X_train_num shape:{self.X_train_num.shape}')
            logger.info(f'X_test_num shape:{self.X_test_num.shape}')
            logger.info(f'X_train_num null values:{self.X_train_num.isnull().sum()}')
            logger.info(f'X_test_num null values:{self.X_test_num.isnull().sum()}')
            # for i in self.X_train['SeniorCitizen']:
            #     if i not in [0,1]:
            #         logger.info(f'{i}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def cat_to_num(self):
        try:
            # logger.info(f'X_train_cat shape:{self.X_train_cat.shape}')
            # logger.info(f'X_test_cat shape:{self.X_test_cat.shape}')
            # logger.info(f'X_train_cat null values:{self.X_train_cat.isnull().sum()}')
            # logger.info(f'X_test_cat null values:{self.X_test_cat.isnull().sum()}')
            self.X_train_cat,self.X_test_cat=CAT_TO_NUMERICAL.categori_to_numeric(self.X_train_cat,self.X_test_cat)
            # logger.info(f'X_train_cat shape:{self.X_train_cat.shape}')
            # logger.info(f'X_test_cat shape:{self.X_test_cat.shape}')
            # logger.info(f'X_train_cat null values:{self.X_train_cat.isnull().sum()}')
            # logger.info(f'X_test_cat null values:{self.X_test_cat.isnull().sum()}')
            # logger.info(f'\n{self.X_train_cat}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def feature_select(self):
        try:
            logger.info(f'{self.X_train_num.isnull().sum()}')
            logger.info(f'{self.X_test_num.isnull().sum()}')
            logger.info(f'{self.X_train_cat.isnull().sum()}')
            logger.info(f'{self.X_test_cat.isnull().sum()}')
            self.X_train_num,self.X_test_num=FEATURE.feature_selecting(self.X_train_num,self.X_test_num,self.y_train)
            self.X_train_num = self.X_train_num.reset_index(drop=True)
            self.X_test_num = self.X_test_num.reset_index(drop=True)
            self.X_train_cat = self.X_train_cat.reset_index(drop=True)
            self.X_test_cat = self.X_test_cat.reset_index(drop=True)
            self.training_data = pd.concat([self.X_train_num, self.X_train_cat], axis=1)
            self.testing_data = pd.concat([self.X_test_num, self.X_test_cat], axis=1)
            logger.info(f'Training data\n{self.training_data.isnull().sum()}')
            logger.info(f'testing data\n{self.testing_data.isnull().sum()}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def data_balance(self):
        try:
            self.training_data,self.y_train=DATA_BALANCE.balancing_data(self.training_data,self.y_train)
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def data_scaling(self):
        try:
            logger.info(f'\n{self.training_data.columns}\n{self.testing_data.columns}')
            self.training_data,self.testing_data,self.y_train,self.y_test=DATA_SCALING.scaling_data(self.training_data,self.testing_data,self.y_train,self.y_test)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def algo(self):
        try:
            logger.info(f'{self.training_data.shape}')
            logger.info(f'{self.testing_data.shape}')
            logger.info(f'{self.y_train.shape}')
            logger.info(f'{self.y_test.shape}')
            TRAINING.all_models(self.training_data,self.testing_data,self.y_train,self.y_test)
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

if __name__ == '__main__':
    try:
        obj=CustomerRetention('D:\\Projects\\Customer churn prediction\\Customer_churn.csv')
        obj.missingvalues()
        obj.variable_transform()
        obj.outliers()
        obj.cat_to_num()
        obj.feature_select()
        obj.data_balance()
        obj.data_scaling()
        obj.algo()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')