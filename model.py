import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


data = pd.read_csv("beer-servings.csv")

if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])

# Handle Missing Values
print(data.isnull().sum())

data = data.dropna()


X = data.drop("total_litres_of_pure_alcohol", axis=1)
y = data["total_litres_of_pure_alcohol"]

categorical_cols = ["country", "continent"]
numeric_cols = ["beer_servings", "spirit_servings", "wine_servings"]


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

lr_model = LinearRegression()

rf_model = RandomForestRegressor(random_state=42)


pipeline_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("model", lr_model)
])

pipeline_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf_model)
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10]
}

grid = GridSearchCV(
    pipeline_rf,
    param_grid,
    cv=5,
    scoring="r2"
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_


pipeline_lr.fit(X_train, y_train)


rf_pred = best_rf.predict(X_test)
lr_pred = pipeline_lr.predict(X_test)


# Evaluate Models
rf_r2 = r2_score(y_test, rf_pred)
lr_r2 = r2_score(y_test, lr_pred)

print("Random Forest R2 Score:", rf_r2)
print("Linear Regression R2 Score:", lr_r2)


if rf_r2 > lr_r2:
    best_model = best_rf
    print("Random Forest Selected")
else:
    best_model = pipeline_lr
    print("Linear Regression Selected")


pickle.dump(best_model, open("model.pkl", "wb"))

print("Model saved successfully as model.pkl")