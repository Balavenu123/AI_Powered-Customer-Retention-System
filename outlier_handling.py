import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('outlier_handling')

class OUTLIER:

    def count_outliers(series):
        Q1=series.quantile(0.25)
        Q3=series.quantile(0.75)
        IQR=Q3-Q1
        lower=Q1-1.5*IQR
        upper=Q3+1.5*IQR
        return ((series < lower) | (series > upper)).sum()

    def outlier_handle(X_train,X_test):
        try:
            columns=['SeniorCitizen','MonthlyCharges','TotalCharges']
            logger.info(f'\n Columns selected for transformation: {columns}')
            folder_path='D:\\Projects\\Customer churn prediction\\outlier_plots'
            for col in columns:
                logger.info(f'Evaluating column: {col}')
                plt.figure()
                sns.boxplot(x=X_train[col])
                plt.savefig(f"{folder_path}/{col}_before_outlier.png", dpi=300, bbox_inches="tight")
                plt.close()

                # Q1=X_train[col].quantile(0.25)
                # Q3=X_train[col].quantile(0.75)
                # IQR=Q3-Q1
                # lower_limit=Q1-1.5*IQR
                # upper_limit=Q3+1.5*IQR
                # outliers=X_train[(X_train[col]<lower_limit)|(X_train[col]>upper_limit)][col]
                # logger.info(f'\n{outliers}')
                # no_of_outliers=len(outliers)
                # logger.info(f'Column {col} has {no_of_outliers} outliers')
                # if no_of_outliers>0:
                #     logger.info(f'Outlier bounds for {col}:{lower_limit},{upper_limit}')

                methods={}
                outlier_counts ={}

                '''For IQR method'''
                Q1=X_train[col].quantile(0.25)
                Q3=X_train[col].quantile(0.75)
                IQR=Q3-Q1
                lower_iqr=Q1-1.5*IQR
                upper_iqr=Q3+1.5*IQR
                s=X_train[col].clip(lower_iqr,upper_iqr)
                methods['IQR']=s
                outlier_counts['IQR']=OUTLIER.count_outliers(s)

                '''for Winsorization'''
                low=X_train[col].quantile(0.05)
                high=X_train[col].quantile(0.95)
                s=X_train[col].clip(low,high)
                methods['Winsorization']=s
                outlier_counts['Winsorization']=OUTLIER.count_outliers(s)

                '''for Quantile'''
                low=X_train[col].quantile(0.01)
                high=X_train[col].quantile(0.99)
                s=X_train[col].clip(low,high)
                methods['Quantile']=s
                outlier_counts['Quantile']=OUTLIER.count_outliers(s)

                '''for Z-score'''
                mean=X_train[col].mean()
                std=X_train[col].std()
                lower_z=mean-3*std
                upper_z=mean+3*std
                s=X_train[col].clip(lower_z,upper_z)
                methods['Z_score']=s
                outlier_counts['Z_score']=OUTLIER.count_outliers(s)

                '''for MAD'''
                median=X_train[col].median()
                mad=np.median(np.abs(X_train[col] - median))
                lower_mad=median-3*mad
                upper_mad=median+3*mad
                s=X_train[col].clip(lower_mad,upper_mad)
                methods['MAD']=s
                outlier_counts['MAD']=OUTLIER.count_outliers(s)

                for method,count in outlier_counts.items():
                    logger.info(f'{method} Remaining outliers: {count}')

                best_method=min(outlier_counts,key=outlier_counts.get)
                logger.info(f'BEST METHOD for {col}: {best_method}')

                X_train[col]=methods[best_method]

                if best_method=='IQR':
                    X_test[col]=X_test[col].clip(lower_iqr,upper_iqr)

                elif best_method=='Winsorization':
                    X_test[col]=X_test[col].clip(X_train[col].quantile(0.05),X_train[col].quantile(0.95))

                elif best_method=='Quantile':
                    X_test[col]=X_test[col].clip(X_train[col].quantile(0.01),X_train[col].quantile(0.99))

                elif best_method=='Z_score':
                    lower_z=mean-3*std
                    upper_z=mean+3*std
                    X_test[col]=X_test[col].clip(lower_z,upper_z)

                elif best_method=='MAD':
                    X_test[col]=X_test[col].clip(lower_mad,upper_mad)

                # logger.info(f'\n {methods}')
                # logger.info(f'\n{outlier_counts}')
                plt.figure()
                sns.boxplot(x=X_train[col])
                plt.title(f"{col} - After {best_method}")
                plt.savefig(f"{folder_path}/{col}_after_{best_method}.png", dpi=300, bbox_inches="tight")
                plt.close()

            return X_train,X_test

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')