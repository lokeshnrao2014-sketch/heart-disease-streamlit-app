# Heart Disease Prediction using Machine Learning

## a. Problem Statement
Heart disease is one of the leading causes of death worldwide. Early detection
plays a crucial role in preventing severe outcomes.  
The objective of this project is to build and compare multiple Machine Learning
models to predict whether a patient is likely to have heart disease based on
clinical parameters.

A Streamlit web application is developed to:
- Upload patient data in CSV format
- Select a trained model
- Predict heart disease
- Display evaluation metrics
- Download prediction results

---

## b. Dataset Description
The dataset used is the Heart Disease dataset.  
Each row represents a patient and contains the following features:

| Feature Name | Description |
|--------------|-------------|
| age | Age of the patient |
| sex | Gender (1 = male, 0 = female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels |
| thal | Thalassemia |
| target | 0 = No Heart Disease, 1 = Heart Disease |

The dataset is preprocessed using:
- Missing value handling  
- Feature scaling  
- Feature engineering:
  - `chol_age_ratio`
  - `high_risk` (based on BP and cholesterol)

---

## c. Models Used and Evaluation

The following six models were trained and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

Each model is evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- MCC (Matthews Correlation Coefficient)  
- AUC Score  

### Comparison Table

| Model                | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) | MCC (%) | AUC (%) |
|----------------------|--------------|---------------|------------|--------------|---------|---------|
| Logistic Regression  | 80.98        | 76.19         | 91.43      | 83.12        | 63.09   | 92.87   |
| Decision Tree        | 98.54        | 100.00        | 97.14      | 98.55        | 97.12   | 98.57   |
| KNN                  | 84.88        | 87.00         | 82.86      | 84.88        | 69.86   | 95.65   |
| Naive Bayes          | 82.44        | 79.49         | 88.57      | 83.78        | 65.21   | 90.15   |
| Random Forest        | 100.00       | 100.00        | 100.00     | 100.00       | 100.00  | 100.00  |
| XGBoost              | 100.00       | 100.00        | 100.00     | 100.00       | 100.00  | 100.00  |


### Observations on Model Performance

- **Logistic Regression**  
  Performs well as a baseline model with **80.98% accuracy** and a strong **AUC of 92.87%**.  
  The high recall (**91.43%**) indicates that it is effective in identifying patients with heart disease.  
  However, its lower precision shows that it may produce more false positives.  
  It is simple, fast, and suitable for real-world deployment.

- **Decision Tree**  
  Achieves **98.54% accuracy** with very high scores across all metrics.  
  While this indicates strong learning capability, such near-perfect performance suggests a risk of **overfitting**, especially on smaller datasets.  
  Decision Trees can easily memorize data without proper pruning.

- **KNN (K-Nearest Neighbors)**  
  Provides balanced performance with **84.88% accuracy** and a high **AUC of 95.65%**.  
  Precision and recall are well balanced, making it a stable classifier.  
  However, KNN is sensitive to feature scaling and may become slow for large datasets.

- **Naive Bayes**  
  Shows decent performance with **82.44% accuracy**.  
  It is computationally efficient and easy to implement, but slightly weaker than Logistic Regression and KNN.  
  Its assumption of feature independence limits its effectiveness for medical data.

- **Random Forest**  
  Achieves **100% in all evaluation metrics**.  
  This demonstrates the strong predictive power of ensemble methods.  
  However, perfect scores strongly indicate **overfitting**, meaning the model may not generalize well to unseen real-world data.

- **XGBoost**  
  Also attains **100% across all metrics**, highlighting its ability to capture complex patterns in structured data.  
  Like Random Forest, such perfect results raise concerns of overfitting and emphasize the need for cross-validation.

**Overall Conclusion:**  
Simpler models such as Logistic Regression provide stable and interpretable results.  
More complex models like Random Forest and XGBoost achieve extremely high performance but require careful validation to ensure generalization.  
This comparison clearly demonstrates how model complexity impacts both accuracy and robustness.


This project demonstrates how multiple machine learning models can be used to solve a real-world healthcare problem and how a web-based interface can be built for real-time prediction and evaluation.
