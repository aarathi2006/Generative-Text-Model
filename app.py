from flask import Flask, render_template, request
from generate import generate_text

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    user_input = ""
    generated_text = ""

    if request.method == "POST":
        user_input = request.form["seed"]
        generated_text = generate_text(user_input)

    return render_template(
        "index.html",
        input_text=user_input,
        output=generated_text
    )

if __name__ == "__main__":
    app.run(debug=True)

