# import flask
from ngb_chatbot import main

from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# API Entrypoint
@app.route('/', methods=['GET','POST'])
@cross_origin()
def chatbot():
    if request.method == 'GET':
        return f'Get request', 200

    req = request.get_json()
    if req and 'message' in req:
        message = req['message']
    else:
        return 'Bad request', 400

    try:
        response = main(message)
    except Exception as err:
        print(f'Error: {err}')
        return f'ChatBot crashed', 400

    if response is None:
        print('No response could be generated')
        return 'No response generated', 400
    else:
        print('Response:', response)
        return f'{response}', 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
