import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── 1. SETUP ──────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("retail-demand-forecasting")
client = MlflowClient()
model_name = "retail-demand-forecasting-model"

# ── 2. DATA GENERATION ────────────────────────────────────────────────────────
def generate_data(n=10000):
    np.random.seed(42)
    data = pd.DataFrame({
        'day_of_week': np.random.randint(0, 7, n),
        'promo_active': np.random.randint(0, 2, n),
        'dept_id': np.random.randint(1, 50, n),
        'inventory_level': np.random.uniform(10, 1000, n)
    })
    # Target: Sales (influenced by features + some noise)
    data['sales'] = (data['promo_active'] * 50) + (data['inventory_level'] * 0.1) + np.random.normal(0, 5, n)
    return data

df = generate_data()
X = df.drop('sales', axis=1)
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 3. THE TOURNAMENT (Hyperparameter Sweep) ──────────────────────────────────
param_grid = [
    {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100},
    {"max_depth": 6, "learning_rate": 0.01, "n_estimators": 200},
    {"max_depth": 10, "learning_rate": 0.05, "n_estimators": 150}
]

print("🚀 Starting model tournament...")

for params in param_grid:
    with mlflow.start_run(run_name=f"xgb-depth-{params['max_depth']}"):
        
        # Train
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics({"rmse": rmse})
        
        # Log the actual model artifact
        mlflow.xgboost.log_model(model, artifact_path="forecast-model")
        
        print(f"Finished: Depth={params['max_depth']} | RMSE={rmse:.4f}")

# ── 4. AUTOMATED CHAMPION PROMOTION ──────────────────────────────────────────
print("\n🔍 Analyzing results to find the Champion...")

# Search for the best run in the experiment based on lowest RMSE
experiment = client.get_experiment_by_name("retail-demand-forecasting")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse ASC"], 
    max_results=1
)

best_run = runs[0]
best_run_id = best_run.info.run_id
best_rmse = best_run.data.metrics['rmse']

print(f"🏆 Best Run Found: {best_run_id} (RMSE: {best_rmse:.4f})")

# Register the model version in the Model Registry
model_uri = f"runs:/{best_run_id}/forecast-model"
mv = mlflow.register_model(model_uri, model_name)

# Assign the 'champion' alias to this version
client.set_registered_model_alias(model_name, "champion", mv.version)

print(f"✅ Version {mv.version} of '{model_name}' is now the 'champion'!")