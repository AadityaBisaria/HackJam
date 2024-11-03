import sys
print(sys.executable)

from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# @app.route("/get")
# def get_bot_response():
#     userText = request.args.get('msg')
#     return 'Hi'
# app.run(debug = True)

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    
    # Run query_data.py with the user message as an argument
    try:
        result = subprocess.run(
            [sys.executable, 'query_data.py', userText],  # Adjust the path if necessary
            capture_output=True,
            text=True,
            check=True
        )
        response = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        response = "Error: " + e.stderr.strip()
    
    return response

app.run(debug = True)
