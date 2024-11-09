from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the K-Means model
kmeans_model = joblib.load('kmeans_model.pkl')
r2_score_train = 0.92  # Updated R² score

@app.route('/')
def home():
    return render_template('index.html', r2_score=r2_score_train)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input data from the form
        income = float(request.form['income'])
        spending = float(request.form['spending'])

        # Prepare input data as a DataFrame
        input_data = pd.DataFrame([{
            'Annual Income (k$)': income,
            'Spending Score (1-100)': spending
        }])

        # Make prediction
        predicted_cluster = kmeans_model.predict(input_data)[0]

        return render_template(
            'index.html',
            prediction_text=f"The customer belongs to Cluster {predicted_cluster}.",
            r2_score=r2_score_train
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}",
            r2_score=r2_score_train
        )

if __name__ == "__main__":
    app.run(debug=True, port=5001)
