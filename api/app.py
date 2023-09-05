from flask import Flask#, render_template, jsonify
# import json
# import requests

DEBUG = True

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


@app.route('/dashboard/')
def dashboard():
    return "the dashboard yo"


@app.route('/api/model')
def model():
    return None


if __name__ == "__main__":
    app.run(debug=DEBUG)