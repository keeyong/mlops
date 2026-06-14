import requests
import json

def predict_sentiment(texts: list[str]) -> dict:
    """Send review texts to the MLflow model server and get predictions."""
    response = requests.post(
        url="http://localhost:5002/invocations", 
        headers={"Content-Type": "application/json"},
        data=json.dumps({"inputs": texts})
    )
    return response.json()

# Example call (requires server to be running)
results = predict_sentiment([
    "This movie was amazing, I loved it!",
    "Terrible film, complete waste of time.",
])
print(results)

