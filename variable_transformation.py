import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import skew,kurtosis,boxcox,yeojohnson
from sklearn.preprocessing import QuantileTransformer
from log_code import setup_logging
logger = setup_logging('variable_transformation')

class VARIABLE_TRANSFORM:
    def variable_trans(X_train, X_test):
        try:
            columns=['tenure','MonthlyCharges','TotalCharges']
            logger.info(f'\n Columns selected for transformation: {columns}')

            plot_folder='D:\\Projects\\Customer churn prediction\\var_transform_plots'
            for col in columns:
                logger.info(f'Evaluating column: {col}')

                plt.figure()
                X_train[col].plot(kind='kde')
                plt.savefig(f"{plot_folder}/{col}_before.png", dpi=300, bbox_inches="tight")
                plt.close()

                x=X_train[col].values.reshape(-1,1)
                scores=[]
                def score_fn(arr):
                    arr=np.asarray(arr).flatten()
                    arr=arr[np.isfinite(arr)]
                    return abs(skew(arr))
                candidates={}
                if (X_train[col]>0).all():
                    candidates['log']=np.log(X_train[col])
                    candidates['reciprocal']=1/X_train[col]
                    candidates['sqrt']=np.sqrt(X_train[col])
                    candidates['inverse_sqrt']=1/np.sqrt(X_train[col])
                candidates['exp']=np.exp(x/X_train[col].max())
                candidates['yeo_johnson'],lam1=stats.yeojohnson(X_train[col])
                if (X_train[col] > 0).all():
                    candidates['boxcox'],lam2=stats.boxcox(X_train[col])
                candidates['quantile']=QuantileTransformer(output_distribution='normal',random_state=42).fit_transform(x)

                for name,arr in candidates.items():
                    scores.append((name,score_fn(arr)))
                score_df=pd.DataFrame(scores,columns=['method','score'])
                best_method=score_df.sort_values('score').iloc[0]['method']
                logger.info(f'BEST METHOD for {col}: {best_method}')

                if best_method=='log':
                    X_train[col]=np.log(X_train[col])
                    X_test[col]=np.log(X_test[col])

                elif best_method=='reciprocal':
                    X_train[col]=1/X_train[col]
                    X_test[col]=1/X_test[col]

                elif best_method=='sqrt':
                    X_train[col]=np.sqrt(X_train[col])
                    X_test[col]=np.sqrt(X_test[col])

                elif best_method=='inverse_sqrt':
                    X_train[col]=1/np.sqrt(X_train[col])
                    X_test[col]=1/np.sqrt(X_test[col])

                elif best_method=='exp':
                    X_train[col]=np.exp(X_train[col]/X_train[col].max())
                    X_test[col]=np.exp(X_test[col]/X_train[col].max())

                elif best_method=='boxcox':
                    X_train[col],lam3=stats.boxcox(X_train[col])
                    logger.info(f'boxcox: {lam3}')
                    X_test[col]=stats.boxcox(X_test[col],lam3)

                elif best_method=='yeo_johnson':
                    X_train[col],lam4=stats.yeojohnson(X_train[col])
                    logger.info(f'yeo_johnson: {lam4}')
                    X_test[col]=stats.yeojohnson(X_test[col],lam4)

                elif best_method=='quantile':
                    qt=QuantileTransformer(output_distribution='normal',random_state=42)
                    qt.fit(X_train[[col]])
                    X_train[col]=qt.transform(X_train[[col]])
                    X_test[col]=qt.transform(X_test[[col]])

                plt.figure()
                X_train[col].plot(kind='kde')
                plt.savefig(f"{plot_folder}/{col}_after_{best_method}.png", dpi=300, bbox_inches="tight")
                plt.close()
                # print(score_df)
            return X_train,X_test
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line no:{error_line.tb_lineno} due to:{error_msg}")
