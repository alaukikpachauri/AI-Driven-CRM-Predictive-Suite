
# 📊 AI CRM Intelligence Suite

**AI-Powered CRM Dashboard for Lead Scoring, Opportunity Win Prediction, and Churn Risk Analysis using XGBoost and Streamlit**

---

## 🚀 Project Overview

This project is a complete end-to-end AI solution for Customer Relationship Management (CRM) analytics. It uses machine learning (XGBoost) to provide predictive insights on:

- ✅ **Lead Scoring** – Predict likelihood of lead conversion.
- ✅ **Opportunity Win Prediction** – Estimate the success rate of opportunities.
- ✅ **Churn Risk** – (New!) Identify accounts at risk of leaving or disengaging.

Built with a simple and interactive **Streamlit interface**, this application empowers sales, marketing, and management teams to make informed decisions with minimal effort.

---

## 🧠 Key Features

- 📂 Upload support for CSV or SQL (`.sql`) files
- 🧮 Built-in preprocessing (imputation, encoding, scaling)
- 📊 Dynamic visualization of model tuning results
- 🔁 Baseline vs Tuned model comparison using ROC AUC
- 💾 Model saving and reloading using `joblib`
- 🛠️ Developer mode for inspecting data types, nulls, and columns
- 🌍 Works with Leads, Opportunities, and Account datasets

---

## 🛠️ Tech Stack

| Component       | Technology             |
|----------------|------------------------|
| UI              | Streamlit              |
| ML Model        | XGBoost + scikit-learn |
| Data Handling   | pandas, numpy          |
| Visualization   | plotly, seaborn        |
| Deployment Ready| Streamlit Cloud / Docker |

---

## 🗂️ Folder Structure

```
ai-crm-insights-xgboost/
│
├── final2.py              # Main Streamlit application
├── requirements.txt       # All Python dependencies
├── README.md              # Project documentation
│
├── /data
│   ├── leads.sql
│   ├── opportunities.sql
│   └── accounts.sql
│
├── /models
│   └── saved_model.joblib
```

---

## ⚙️ How to Run

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-crm-insights-xgboost.git
cd ai-crm-insights-xgboost
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**
```bash
streamlit run final2.py
```

4. **Upload CRM data** as `.sql` or `.csv` to begin scoring!

---

## 📷 Screenshots

_Add screenshots or a demo video link here._

---

## 📈 Sample Use Cases

- Sales team: Focus on leads most likely to convert.
- Managers: Review win-rate of current deals.
- Support team: Detect accounts with churn risk.
- Executives: Visualize data trends across CRM records.

---

## 📌 Future Roadmap

- 🔐 Add user authentication
- 🌐 Add REST API endpoints
- 🔄 Scheduled retraining (cron jobs)
- 🧠 Model Explainability with SHAP

---

## 🙋‍♂️ Author

**Alaukik Pachauri**  
[LinkedIn](https://linkedin.com/in/yourprofile) • [Email](mailto:your@email.com)

---

## 📝 License

This project is licensed under the MIT License.
