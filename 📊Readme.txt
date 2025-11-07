# 📊 Telecom Customer Churn Prediction Project

This project analyzes customer churn data for a telecom company and builds a **machine learning model** to predict which customers are likely to leave. It also provides **KPI dashboards** and a **business insights report** for retention strategy.

------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🚀 Project Overview

### 🎯 Objective
To understand why customers churn and predict who will leave next month using machine learning.  
This helps the telecom company take **proactive actions** — such as offering discounts, improving service, or providing better customer support.

------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🧠 Machine Learning Model

- **Model Used:** Random Forest Classifier  
- **Accuracy:** 78.48%  
- **Precision (Churn):** 62%  
- **Recall (Churn):** 49%  
- **F1-Score:** 54.8%

✅ The model correctly predicts ~4 out of 5 customers overall.  
It’s strong at identifying who will stay and provides good early warning signals for churners.

-----------------------------------------------------------------------------------------------------------------------------------------------------------

## 📈 Key Insights

| Feature | Impact on Churn |
|----------|-----------------|
| TotalCharges | Customers paying more overall tend to leave more often |
| Tenure | New customers are more likely to churn |
| MonthlyCharges | Higher monthly bills increase churn probability |
| Contract (Month-to-Month) | Short-term users are least loyal |
| TechSupport_No | Customers without support are more likely to leave |

------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📊 KPI Dashboard

The dashboard visualizes major churn insights:
1. **% of Customers Left vs Stayed** – 26.6% customers left.
2. **Churn by Contract Type** – Month-to-month plans have the highest churn (42.7%).
3. **Churn by Internet Service Type** – Fiber optic users churn the most (41.9%).
4. **Top Features Influencing Churn** – Displays top 5 churn drivers in a pie chart.

All charts are generated using Matplotlib and automatically displayed in a 2x2 KPI grid.

------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📘 Report

A detailed **PDF Report** (`Telecom_Customer_Churn_Report.pdf`) is automatically created, containing:
- Model evaluation results  
- KPI explanations  
- Business insights  
- Recommendations to reduce churn  
- Embedded charts and visuals  

------------------------------------------------------------------------------------------------------------------------------------------------------------

## 💡 Recommendations

- Offer **discounts** to long-tenure and high-billing customers.  
- Promote **yearly contracts** with added benefits.  
- Provide **free tech support** for new or month-to-month users.  
- Monitor **Fiber Optic** complaints and service quality.  
- Use this churn model monthly for targeted retention campaigns.

------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🧰 Project Structure

📁 Telecom_Customer_Churn_Project
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset
├── telecom_model.py # Main Python script
├── output.png # Model output metrics image
├── telco_kpi_dashboard.png # KPI dashboard visualization
├── Telecom_Customer_Churn_Report.pdf # Final PDF report
├── requirements.txt # Required dependencies
└── README.md # Project documentation

➡️To install all dependencies:

pip install -r requirements.txt


------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🧩 How to Run the Project

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the main script
python telecom_model.py

# Step 3: View generated outputs
# - Charts open automatically
# - PDF Report saved as Telecom_Customer_Churn_Report.pdf
