# Used Car Price Predictor

![Used Car Image](image%20copy.png)  

## **Project Overview**
This project predicts used car prices using machine learning. It considers features like make, model, year, mileage, and condition to estimate fair market values, benefiting both buyers and sellers.

## **Features**

### **Data Collection & Preprocessing:**
- Data sourced from **Kaggle** and **Web Scraping**.
- Data cleaning included handling missing values, outliers, and converting categorical features.

### **Feature Engineering:**
- Engineered features like **vehicle age, mileage brackets, and brand popularity**.

### **Model Building:**
- Trained models: **Linear Regression, Random Forest, Gradient Boosting**.
- Model evaluation using **R-squared, MSE, and RMSE**.

### **Model Selection & Tuning:**
- Selected the best model via **hyperparameter tuning**.

### **Deployment:**
- Deployed as a **web app using Streamlit**.

## **MLflow Experiment Tracking**
We used **MLflow** to track different model versions and performance metrics. The best model (Champion) was selected based on evaluation metrics.

### **MLruns Artifacts & Model Tracking**
![MLruns Artifacts](mlruns_artifacts.png)
![MLflow Model Comparison](mlruns_comparison.png)

### **Champion vs. Challenger Models**
| Model | R-Squared | MSE | RMSE |
|--------|------------|------------|------------|
| **Random Forest (Champion)** | **0.8021** | **1.16e+11** | **340989.16** |
| Linear Regression (Challenger) | 0.7324 | 1.48e+11 | 385417.89 |
| Gradient Boosting (Challenger) | 0.7856 | 1.22e+11 | 349232.45 |

## **Technologies Used**
- **Languages:** Python
- **ML Libraries:** scikit-learn, XGBoost
- **Data Libraries:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Web Framework:** Flask, Streamlit
- **Experiment Tracking:** MLflow

## **How to Use**

1. **Clone:** `git clone https://github.com/Harshit-Chaudhry/Used-Car-Price-Predictor.git`
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Run the App:** `streamlit run app.py`
4. **Access Web App:** `http://127.0.0.1:5000/`

## **Results & Evaluation**
- **Best Model:** Random Forest Regressor
- **R-squared:** **0.8021**  
- **MSE:** **1.16e+11**  
- **RMSE:** **340989.16**  

## **Future Work**
- Explore deep learning models.
- Build an **AI Agent** for the automobile industry.

## **Contributing**
Contributions are welcome! Open issues or submit pull requests.

## **Contact**
**Harshit Chaudhary**  
📧 Email: Chaudhary.harshit1203@gmail.com  
🔗 LinkedIn: [Harshit Chaudhary](https://www.linkedin.com/in/harshit-chaudhary-Ai-Agent/)
