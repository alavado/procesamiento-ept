from flask import Flask, request
app = Flask(__name__)

@app.route('/')
def hello_world():
  return 'Hello, World!'

@app.route('/proc', methods=['POST'])
def proc():
  print(request.form['id'])
  return 'aasx'

app.run(debug=True)