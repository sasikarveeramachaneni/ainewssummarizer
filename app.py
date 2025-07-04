from flask import Flask, render_template, request
from main import summarize_url  # Make sure this function is defined in main.py

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        url = request.form.get("url")
        if url:
            summary = summarize_url(url)
    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
