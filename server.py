from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
  return 'Hello, World!'

@app.route('/proc', methods=['POST'])
def proc():
  return 'aasx'

app.run(debug=True)