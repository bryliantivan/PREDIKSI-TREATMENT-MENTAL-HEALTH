from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the Stacking Model (replace with your model file path)
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Route for landing page
@app.route("/")
def landing():
    return render_template("landing.html")  # Serve the landing page first

# Route for index page where the prediction form is
@app.route("/index")
def home():
    return render_template("index.html")  # This will be the page for predictions

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Convert categorical features to numerical values for the model
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    family_history_map = {"Yes": 1, "No": 0}
    benefits_map = {"Yes": 1, "No": 0}
    care_options_map = {"Yes": 1, "No": 0}
    anonymity_map = {"Yes": 1, "No": 0}
    leave_map = {"Easy": 1, "Difficult": 2, "Don't know": 0}
    work_interfere_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}

    # Prepare the features for the model prediction
    features = np.array([[ 
        data["age"],
        gender_map.get(data["gender"], -1),
        family_history_map.get(data["family_history"], -1),
        benefits_map.get(data["benefits"], -1),
        care_options_map.get(data["care_options"], -1),
        anonymity_map.get(data["anonymity"], -1),
        leave_map.get(data["leave"], -1),
        work_interfere_map.get(data["work_interfere"], -1)
    ]])

    # Make the prediction using the loaded model
    prediction = model.predict(features)[0]
    
    # Send the prediction back as a response
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
