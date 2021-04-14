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
  ti = request.form['ti']
  tf = request.form['tf']
  c = request.form['canal']
  tir = request.form['tir']
  tfr = request.form['tfr']
  cr = request.form['canalr']
  return jsonify(processAll(imu, emg, ti, tf, c, tir, tfr, cr))

app.run()