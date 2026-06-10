from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load("random_forest_model.pkl")

# Home Page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = data["features"]

        # Make prediction
        prediction = model.predict([features])[0]

        # Convert class number to flower name
        flowers = {
            0: "Iris Setosa",
            1: "Iris Versicolor",
            2: "Iris Virginica"
        }

        result = flowers.get(prediction, "Unknown")

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)