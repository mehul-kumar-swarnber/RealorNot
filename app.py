from flask import Flask, render_template, request, session
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session tracking

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if "recent_news" not in session:
        session["recent_news"] = []

    if request.method == "POST":
        text = request.form["news"]
        vec = vectorizer.transform([text])
        result = model.predict(vec)
        prediction = "Real News 🟢" if result[0] == 1 else "Fake News 🔴"

        # Add to recent searches (limit to 5 items)
        recent_news = session["recent_news"]
        recent_news.insert(0, text)
        session["recent_news"] = recent_news[:5]

    return render_template("index.html", prediction=prediction, recent_news=session["recent_news"])

if __name__ == "__main__":
    app.run(debug=True)
