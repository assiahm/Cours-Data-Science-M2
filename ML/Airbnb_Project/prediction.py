# Airbnb Price Prediction - Projet Machine Learning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 📥 Chargement des données
train = pd.read_csv("airbnb_train.csv")
test = pd.read_csv("airbnb_test.csv")

# 🌟 Séparation de la cible
y = train["log_price"]
X = train.drop(columns=["log_price", "id"])
test_ids = test["Unnamed: 0"]
X_test = test.drop(columns=["Unnamed: 0"])

# 🧹 Sélection de colonnes pertinentes
selected_cat_cols = [
    "property_type", "room_type", "bed_type", "cancellation_policy",
    "city", "instant_bookable", "cleaning_fee"
]

selected_num_cols = [
    "accommodates", "bathrooms", "bedrooms", "beds",
    "number_of_reviews", "review_scores_rating", "latitude", "longitude"
]

# 🔧 Pipelines de prétraitement
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, selected_num_cols),
    ("cat", cat_pipeline, selected_cat_cols)
])

# 🧠 Modèle RandomForest
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# 🧪 Split train / validation
X_train, X_val, y_train, y_val = train_test_split(
    X[selected_num_cols + selected_cat_cols], y, test_size=0.2, random_state=42
)

# 🚀 Entraînement
model.fit(X_train, y_train)

# 📏 Évaluation
val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print(f"Validation RMSE: {rmse:.4f}")

# 📊 Visualisation des erreurs
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_val, y=val_preds, alpha=0.5)
plt.xlabel("Vrai log_price")
plt.ylabel("Prédit log_price")
plt.title("Prédictions vs Réalité")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.tight_layout()
plt.show()

# 📁 Prédiction sur les données test
test_preds = model.predict(X_test[selected_num_cols + selected_cat_cols])

# 📆 Sauvegarde des résultats
submission = pd.DataFrame({
    "id": test_ids,
    "prediction": test_preds
})
submission.to_csv("prediction.csv", index=False)
print("✅ Fichier prediction.csv généré.")
