from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# DAG definition
default_args = {"start_date": datetime(2023, 1, 1)}
dag = DAG(
    "iris_ml_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["ml", "demo"],
)

# Step 1: Load dataset
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df.to_csv("/tmp/iris.csv", index=False)

# Step 2: Train model
def train_model():
    df = pd.read_csv("/tmp/iris.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    pd.DataFrame(clf.predict(X_test)).to_csv("/tmp/predictions.csv", index=False)

# Step 3: Evaluate model
def evaluate_model():
    df = pd.read_csv("/tmp/iris.csv")
    y_true = df["target"].tail(30).values  # last 30 samples
    y_pred = pd.read_csv("/tmp/predictions.csv").values.flatten()
    acc = accuracy_score(y_true[:len(y_pred)], y_pred)
    print(f"Model Accuracy: {acc:.2f}")

# Tasks
load_task = PythonOperator(task_id="load_data", python_callable=load_data, dag=dag)
train_task = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)
eval_task = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model, dag=dag)

load_task >> train_task >> eval_task
