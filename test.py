from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("customer_churn_model.json")