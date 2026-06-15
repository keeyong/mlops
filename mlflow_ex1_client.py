import requests

MODEL_SERVER_URL = "http://localhost:5002/invocations"


def predict_sentiment(text: str) -> dict:
    """Send exactly one review text to the MLflow model server."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if not text.strip():
        raise ValueError("text must not be empty.")

    response = requests.post(
        MODEL_SERVER_URL,
        headers={"Content-Type": "application/json"},
        json={
            "dataframe_records": [
                {"text": text}
            ]
        },
        timeout=30,
    )

    if not response.ok:
        raise RuntimeError(
            f"MLflow request failed ({response.status_code}): "
            f"{response.text}"
        )

    result = response.json()

    predictions = result.get("predictions", [])
    if len(predictions) != 1:
        raise RuntimeError(
            f"Expected exactly one prediction, but received: {result}"
        )
    return result["predictions"][0]


if __name__ == "__main__":
    reviews = [
        "This movie was amazing, I loved it!",
        "Dull and predictable. Fell asleep halfway through",
        "Too bad"
    ]
    for r in reviews:
        result = predict_sentiment(r)
        print(r, result)
