from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

data = pd.read_csv("beer-servings.csv")

countries = sorted(data["country"].unique())
continents = sorted(data["continent"].unique())


@app.route("/")
def home():
    return render_template(
        "home.html",
        countries=countries,
        continents=continents
    )


@app.route("/predict", methods=["POST"])
def predict():

    country = request.form["country"]
    continent = request.form["continent"]

    beer = float(request.form["beer_servings"])
    spirit = float(request.form["spirit_servings"])
    wine = float(request.form["wine_servings"])

    input_data = pd.DataFrame({
        "country": [country],
        "continent": [continent],
        "beer_servings": [beer],
        "spirit_servings": [spirit],
        "wine_servings": [wine]
    })

    prediction = model.predict(input_data)[0]

    prediction = round(prediction, 2)

    return render_template(
        "home.html",
        prediction_text=f"Predicted Total Litres of Pure Alcohol: {prediction}",
        countries=countries,
        continents=continents
    )


if __name__ == "__main__":
    app.run(debug=True)