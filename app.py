from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    print("FORM DATA RECEIVED:", request.form)

    try:
        features = np.array([[
            float(request.form["age"]),
            int(request.form["sex"]),
            float(request.form["alb"]),
            float(request.form["alp"]),
            float(request.form["alt"]),
            float(request.form["ast"]),
            float(request.form["bil"]),
            float(request.form["che"]),
            float(request.form["chol"]),
            float(request.form["crea"])
        ]])

        pred = model.predict(features)[0]
        print("RAW PREDICTION:", pred)

        return render_template("index.html", prediction=str(pred))

    except Exception as e:
        print("🔥 BACKEND ERROR:", e)
        return render_template("index.html", prediction="ERROR")


if __name__ == "__main__":
    app.run(debug=True)
