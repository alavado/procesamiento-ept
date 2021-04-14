from flask import Flask, request, jsonify
import sys
from todo_ept import processAll
import pandas as pd
app = Flask(__name__)

@app.route('/')
def hello_world():
  return 'Hello, World!'

@app.route('/proc', methods=['POST'])
def proc():
  imu = pd.read_csv(request.files['imu'])
  emg = pd.read_csv(request.files['emg'])
  return jsonify(processAll(imu, emg))

app.run()