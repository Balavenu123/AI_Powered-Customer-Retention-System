<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  
</head>
<body>

<h1>🤖 AI_Powered-Customer-Retention-prediction-System</h1>

<p>
An <strong>end-to-end AI-driven Machine Learning system</strong> to predict
<strong>telecom customer churn</strong> and enable
<strong>proactive customer retention strategies</strong>.
</p>

<hr>

<h2>📌 Project Objective</h2>
<p>
To build an intelligent system that automatically selects the
<strong>best preprocessing techniques, optimal ML model, and deployment-ready pipeline</strong>
to accurately predict customer churn.
</p>

<hr>

<h2>🧠 Complete Machine Learning Pipeline</h2>

<ol>
  <li>Data Loading</li>
  <li>Missing Value Handling</li>
  <li>Variable Transformation</li>
  <li>Outlier Detection & Treatment</li>
  <li>Categorical Encoding</li>
  <li>Feature Selection</li>
  <li>Class Imbalance Handling</li>
  <li>Feature Scaling</li>
  <li>Model Training & Selection</li>
  <li>Hyperparameter Tuning</li>
  <li>Deployment</li>
</ol>

<hr>

<h2>🔍 Preprocessing Techniques (Final Selections)</h2>

<h3>1️⃣ Missing Value Handling</h3>
<p>
Column handled: <strong>TotalCharges</strong>
</p>
<p>
Multiple imputation strategies were evaluated:
</p>
<ul>
  <li>Mean Imputer</li>
  <li>Median Imputer</li>
  <li>Mode Imputer</li>
  <li>Constant & Arbitrary Value</li>
  <li>Random Sampling</li>
  <li>Iterative Imputer</li>
  <li>KNN Imputer</li>
</ul>

<p>
<strong>Final Selected Method:</strong><br>
✔ Selected Random Sampling based on <strong>minimum deviation in mean and standard deviation</strong>.
</p>

<hr>

<h3>2️⃣ Variable Transformation</h3>
<p>
Applied on numeric features:
</p>
<ul>
  <li>tenure</li>
  <li>MonthlyCharges</li>
  <li>TotalCharges</li>
</ul>

<p>
Candidate transformations tested:
</p>
<ul>
  <li>Log</li>
  <li>Square Root</li>
  <li>Reciprocal</li>
  <li>Exponential</li>
  <li>Box-Cox</li>
  <li>Yeo-Johnson</li>
  <li>Quantile Transformer</li>
</ul>

<p>
<strong>Final Selected Method:</strong><br>
✔ Quantile Transformation and Yeo-Johnson are selected based on <strong>lowest skewness score</strong>.
</p>

<hr>

<h3>3️⃣ Outlier Detection & Treatment</h3>
<p>
Handled for:
</p>
<ul>
  <li>SeniorCitizen</li>
  <li>MonthlyCharges</li>
  <li>TotalCharges</li>
</ul>

<p>
Methods evaluated:
</p>
<ul>
  <li>IQR Method</li>
  <li>Winsorization</li>
  <li>Quantile Clipping</li>
  <li>Z-Score</li>
  <li>MAD (Median Absolute Deviation)</li>
</ul>

<p>
<strong>Final Selected Method:</strong><br>
✔ IQR Method is used as it is removing all the outliers.
</p>

<hr>

<h3>4️⃣ Categorical Encoding</h3>

<ul>
  <li><strong>Label Encoding</strong> → Binary variables (Gender, Partner, Dependents, etc.)</li>
  <li><strong>One-Hot Encoding</strong> → Multi-class services (InternetService, StreamingTV, etc.)</li>
  <li><strong>Ordinal Encoding</strong> → Contract (Month-to-month &lt; One year &lt; Two year)</li>
  <li><strong>Frequency Encoding</strong> → PaymentMethod, Sim, Region</li>
</ul>

<p>
<strong>Final Output:</strong><br>
✔ Encoding maps saved as <code>freq_maps.pkl</code> and reused during prediction.
</p>

<hr>

<h3>5️⃣ Feature Selection</h3>

<p>
Techniques applied sequentially:
</p>
<ul>
  <li>Constant feature removal</li>
  <li>Quasi-constant feature removal</li>
  <li>Pearson correlation hypothesis testing</li>
</ul>

<p>
<strong>Final Selected Features:</strong><br>
✔ Only statistically significant and variance-rich features retained.
</p>

<hr>

<h3>6️⃣ Class Imbalance Handling</h3>

<p>
<strong>Technique Used:</strong><br>
✔ <strong>SMOTE (Synthetic Minority Oversampling Technique)</strong>
</p>

<p>
Balances churn vs non-churn classes to improve recall and ROC-AUC.
</p>

<hr>

<h3>7️⃣ Feature Scaling</h3>

<p>
Scalers evaluated:
</p>
<ul>
  <li>StandardScaler</li>
  <li>MinMaxScaler</li>
  <li>RobustScaler</li>
  <li>MaxAbsScaler</li>
</ul>

<p>
<strong>Final Selected Scaler:</strong><br>
✔ StandardScalar is used as final method.
</p>

<p>
Saved as <code>scaler.pkl</code>
</p>

<hr>

<h2>🤖 Model Training & Selection</h2>

<p>
Models trained and evaluated:
</p>
<ul>
  <li>Logistic Regression</li>
  <li>KNN</li>
  <li>Naive Bayes</li>
  <li>Decision Tree</li>
  <li>Random Forest</li>
  <li>AdaBoost</li>
  <li>Gradient Boosting</li>
  <li>XGBoost</li>
  <li>Support Vector Machine (SVM)</li>
</ul>

<p>
<strong>Final Model Selection Criterion:</strong><br>
✔ Highest <strong>ROC-AUC Score I got Logistic Regression as best model</strong>
</p>

<hr>

<h2>⚙️ Hyperparameter Tuning</h2>

<p>
<strong>Technique Used:</strong><br>
✔ <strong>GridSearchCV</strong> with 5-fold cross-validation
</p>

<p>
Final tuned model saved as <code>best_model.pkl</code>
</p>

<hr>

<h2>🌐 Deployment</h2>

<ul>
  <li>Flask backend</li>
  <li>HTML & CSS frontend</li>
  <li>Real-time churn prediction</li>
  <li>Probability score output</li>
</ul>

<p>
Production-ready using:
</p>
<ul>
  <li>Gunicorn</li>
  <li>Procfile</li>
</ul>

<hr>

<h2>👨‍💻 Author</h2>
<p>
<strong>Bala Venu Balineni</strong><br>
AI & Machine Learning Enthusiast
</p>

<p>⭐ If you like this project, give it a star on GitHub!</p>

</body>
</html>
