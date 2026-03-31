# 🔧 Predictive Maintenance Dashboard

A machine learning powered dashboard that predicts machine failures **before they happen**, using the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset).

> Built by **Shail Vaghela** | Mechanical Engineering & Management Graduate  
> 📍 Cologne, Germany | 🐍 Python · Scikit-learn · Streamlit

---

## 🎯 Problem Statement

Unplanned machine downtime costs manufacturers an average of **€250,000 per hour**. Traditional maintenance is either too late (reactive) or wasteful (scheduled). This project demonstrates how machine learning can predict failures in advance — enabling smarter, data-driven maintenance decisions.

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset
- **Size:** 10,000 data points, 14 features
- **Target:** Machine failure (binary: 0 = No failure, 1 = Failure)
- **Failure rate:** ~3.4% (class imbalance handled via `class_weight='balanced'`)

**Features used:**
| Feature | Description |
|---|---|
| Type | Machine type: L (Light), M (Medium), H (Heavy) |
| Air temperature [K] | Ambient air temperature |
| Process temperature [K] | Machine process temperature |
| Rotational speed [rpm] | Spindle rotation speed |
| Torque [Nm] | Applied torque |
| Tool wear [min] | Cumulative tool wear time |

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Pandas & NumPy** — data manipulation
- **Scikit-learn** — Random Forest model
- **Matplotlib & Seaborn** — visualizations
- **Streamlit** — interactive dashboard

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/predictive-maintenance-dashboard.git
cd predictive-maintenance-dashboard
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python model.py
```

**4. Launch the dashboard**
```bash
streamlit run app.py
```

The dashboard opens automatically at `http://localhost:8501`

---

## 📁 Project Structure

```
predictive-maintenance-dashboard/
│
├── app.py                  → Streamlit interactive dashboard
├── model.py                → Model training & evaluation
├── requirements.txt        → Python dependencies
│
├── data/
│   └── predictive_maintenance.csv   → Dataset (auto-downloaded)
│
├── notebooks/
│   └── exploratory_analysis.ipynb  → EDA notebook
│
└── plots/
    ├── confusion_matrix.png         → Model evaluation
    └── feature_importance.png       → Feature analysis
```

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~97% |
| Model | Random Forest Classifier |
| Trees | 100 estimators |
| Class imbalance | Handled via `class_weight='balanced'` |

---

## 🖥️ Dashboard Features

- **Real-time failure prediction** — adjust machine parameters via sliders
- **Failure probability gauge** — visual risk indicator
- **Feature importance chart** — understand what drives failures
- **Dataset explorer** — browse the raw data
- **Failure distribution analysis** — understand class balance

---

## 💡 Key Learnings

- Tool wear and torque are the strongest predictors of machine failure
- Heavy (H) machines have a significantly higher failure rate than L or M types
- Class imbalance must be addressed to avoid biased predictions
- Random Forest outperforms simpler models on this dataset due to non-linear feature interactions

---

## 🔜 Next Steps

- [ ] Add failure type classification (TWF, HDF, PWF, OSF, RNF)
- [ ] Implement real-time data ingestion via IoT simulation
- [ ] Add Carbon footprint estimation per production run
- [ ] Deploy dashboard to Streamlit Cloud

---

## 📬 Contact

**Shail Vaghela**  
📧 shailvaghela0303@gmail.com  
📍 Cologne, Germany  
🔗 [LinkedIn](https://linkedin.com/in/shailvaghela)
