from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

class TitanicInput(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    Embarked: str

app = FastAPI()
model = joblib.load("titanic_pipeline.pkl")

@app.post("/predict")
def predict(data: TitanicInput):
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}


# --- Streamlit section ---
import streamlit as st

st.title("Titanic Survival Prediction")

Pclass = st.selectbox("Pclass", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 100, 25)
Fare = st.slider("Fare", 0.0, 500.0, 32.0)
Embarked = st.selectbox("Embarked", ["S", "C", "Q"])

if st.button("Predict"):
    X_new = pd.DataFrame([[Pclass, Sex, Age, Fare, Embarked]],
                         columns=["Pclass", "Sex", "Age", "Fare", "Embarked"])
    pred = model.predict(X_new)
    st.write("Prediction:", "Survived" if pred[0] == 1 else "Did not survive")