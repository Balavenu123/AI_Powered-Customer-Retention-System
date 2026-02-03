import sys
import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('missing_values')

class MissingValues:
    def null_value_handling(X_train,X_test):
        try:
            col='TotalCharges'
            results=[]
            base_mean=X_train[col].mean() # base mean
            base_std=X_train[col].std()   # base std
            logger.info(f'================base mean and std==============')
            logger.info(f"base Mean: {base_mean}")
            logger.info(f"base Std: {base_std}")

            def evaluate(method, series):
                mean=series.mean()
                std=series.std()
                score=abs(mean-base_mean)+abs(std-base_std)

                results.append({
                    "Method": method,
                    "Mean": mean,
                    "Std": std,
                    "Mean_Diff": abs(mean - base_mean),
                    "Std_Diff": abs(std - base_std),
                    "Score": score
                })

                logger.info(f'{method} | Mean: {mean} | Std: {std} | Score: {score}')

            # Methods to handle null values
            s=X_train[col].fillna(base_mean)  # for mean
            evaluate('Mean Imputer',s)

            median_val=X_train[col].median()  # for median
            s=X_train[col].fillna(median_val)
            evaluate('Median Imputer',s)

            mode_val=X_train[col].mode()[0]   # for mode
            s=X_train[col].fillna(mode_val)
            evaluate('Mode Imputer',s)

            s=X_train[col].fillna(0)          # for constant
            evaluate('Constant(0)',s)

            s=X_train[col].fillna(999999)     # for arbitrary constant
            evaluate('Arbitrary(999999)',s)

            extreme=base_mean+3*base_std  # for end of distribution
            s=X_train[col].fillna(extreme)
            evaluate('End of Distribution',s)

            s=X_train[col].copy()             # for random Sample imputer
            non_null=s.dropna()
            s[s.isnull()]=np.random.choice(non_null,size=s.isnull().sum())
            evaluate('Random Sample',s)

            it=IterativeImputer(random_state=42)  # for iterative Imputer
            it.fit(X_train[[col]])
            it_vals=it.transform(X_train[[col]])
            evaluate('Iterative Imputer',pd.Series(it_vals.flatten()))

            knn=KNNImputer(n_neighbors=5)     # KNN Imputer
            knn.fit(X_train[[col]])
            knn_vals=knn.transform(X_train[[col]])
            evaluate('KNN Imputer',pd.Series(knn_vals.flatten()))

            s=X_train[col].fillna(method='ffill') # for forward Fill
            evaluate('Forward Fill',s)

            s=X_train[col].fillna(method='bfill') # for backward Fill
            evaluate('Backward Fill',s)

            result_df=pd.DataFrame(results).sort_values('Score')
            #logger.info(f"{result_df}")
            best_method=result_df.iloc[0]['Method']
            logger.info(f'best method: {best_method}')

            if best_method=='Mean Imputer':
                X_train[col].fillna(base_mean,inplace=True)
                X_test[col].fillna(base_mean,inplace=True)

            elif best_method=='Median Imputer':
                X_train[col].fillna(median_val,inplace=True)
                X_test[col].fillna(median_val,inplace=True)

            elif best_method=='Mode Imputer':
                X_train[col].fillna(mode_val,inplace=True)
                X_test[col].fillna(mode_val,inplace=True)

            elif best_method=='Constant(0)':
                X_train[col].fillna(0,inplace=True)
                X_test[col].fillna(0,inplace=True)

            elif best_method=='Arbitrary(999999)':
                X_train[col].fillna(999999,inplace=True)
                X_test[col].fillna(999999,inplace=True)

            elif best_method=='End of Distribution':
                X_train[col].fillna(extreme,inplace=True)
                X_test[col].fillna(extreme,inplace=True)

            elif best_method=='Random Sample':
                for df in [X_train, X_test]:
                    a=df[col]
                    not_null=a.dropna()
                    a[a.isnull()]=np.random.choice(not_null,size=a.isnull().sum())

            elif best_method=='Iterative Imputer':
                it=IterativeImputer(random_state=42)
                it.fit(X_train[[col]])
                X_train[col]=it.transform(X_train[[col]])
                X_test[col]=it.transform(X_test[[col]])

            elif best_method=='KNN Imputer':
                knn=KNNImputer(n_neighbors=5)
                knn.fit(X_train[[col]])
                X_train[col]=knn.transform(X_train[[col]])
                X_test[col]=knn.transform(X_test[[col]])

            elif best_method=='Forward Fill':
                X_train[col].fillna(method='ffill',inplace=True)
                X_test[col].fillna(method='ffill',inplace=True)

            elif best_method=='Backward Fill':
                X_train[col].fillna(method='bfill',inplace=True)
                X_test[col].fillna(method='bfill',inplace=True)

            return X_train, X_test

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')