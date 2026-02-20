import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# ==============================
# Load Real Housing Dataset
# ==============================

housing = fetch_california_housing()

data = pd.DataFrame(housing.data, columns=housing.feature_names)
data["price"] = housing.target

print("Dataset Preview:")
print(data.head())


# ==============================
# Features and Target
# ==============================

X = data.drop("price", axis=1)
y = data["price"]


# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# Models
# ==============================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100)
}

results = {}


print("\n===== MODEL TRAINING & COMPARISON =====")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    results[name] = r2

    print(f"\n{name}")
    print("R2 Score:", r2)
    print("Mean Squared Error:", mse)


# ==============================
# Best Model Selection
# ==============================

best_model_name = max(results, key=results.get)
print("\n Best Model:", best_model_name)


# ==============================
# Visualization: Model Comparison
# ==============================

plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values())
plt.title("Model Comparison (R2 Score)")
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.show()


# ==============================
# Feature Importance (Random Forest)
# ==============================

rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)

importance = rf_model.feature_importances_

plt.figure(figsize=(8, 5))
plt.bar(X.columns, importance)
plt.xticks(rotation=45)
plt.title("Feature Importance (Random Forest)")
plt.show()


# ==============================
# Actual vs Predicted Graph (Best Model)
# ==============================

best_model = models[best_model_name]
best_model.fit(X_train, y_train)
pred_test = best_model.predict(X_test)

plt.scatter(y_test, pred_test)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Actual vs Predicted ({best_model_name})")
plt.show()