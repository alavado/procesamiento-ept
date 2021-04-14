from flask import Flask, request, jsonify
import sys
from todo_ept import processAll
import pandas as pd
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)

@app.route('/')
def hello_world():
  return 'Hello, World!'

@app.route('/proc', methods=['POST'])
@cross_origin()
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