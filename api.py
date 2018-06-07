import datetime
from flask import Flask, request, jsonify


app = Flask(__name__)
STATE = {'mode': 0, 'selection':0}

@app.route('/state', methods=['GET'])
def get_state():
    return jsonify(STATE)

@app.route('/state', methods=['POST'])
def update_state():
    instruction = request.get_json()
    if instruction['mode'] is not None:
        STATE['mode'] = instruction['mode']
    if instruction['selection'] is not None:
        STATE['selection'] = instruction['selection']
    return jsonify({'message': 'State updated!'})

if __name__ == '__main__':
    app.run(debug=True)
