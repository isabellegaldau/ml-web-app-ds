
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model_file = "/workspaces/ml-web-app-ds/models/random_forest_model.pkl"
class_dict = {
    0: "Normal Weight",
    1: "Overweight Level I",
    2: "Overweight Level II",
    3: "Overweight Level III",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III"
}

# Load the trained model
with open(model_file, 'rb') as file:
    model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the input values from the form
        age = float(request.form["age"])
        gender = float(request.form["gender"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        fcvc = float(request.form["fcvc"])
        ncp = float(request.form["ncp"])

        # Make prediction
        prediction = model.predict([[age, gender, height, weight, fcvc, ncp]])
        pred_class = class_dict[prediction[0]]
    else:
        pred_class = None
    
    return render_template("index.html", prediction=pred_class)


