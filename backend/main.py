from ngb_chatbot import main
# from sentence_transformers import SentenceTransformer
import spacy
import numpy as np
import pandas as pd
import sys

from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
# from flask_sockets import Sockets

app = Flask(__name__)
# sockets = Sockets(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


nlp = spacy.load('./en_core_web_md-2.3.1')
# model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
df_data = pd.read_csv(f'./ngb_data.csv', encoding='utf8')
sent_emb = np.load(f'./sent_emb_gloveTFIDF.npy', allow_pickle=True)

# API Entrypoint
@app.route('/', methods=['GET','POST'])
@cross_origin()
def chatbot():
    if request.method == 'GET':
        # return render_template('index.html')
        return f'Get request', 200

    req = request.get_json()
    if req and 'message' in req:
        message = req['message']
    else:
        return 'Bad request', 400

    try:
        response = main(nlp, df_data, sent_emb, message)
    except Exception as err:
        print(f'Error: {err}')
        return f'ChatBot crashed', 400

    if response is None:
        print('No response could be generated')
        return 'No response generated', 400
    else:
        print('Response:', response)
        return f'{response}', 200


# @sockets.route('/chat')
# def chat(ws):
#     while not ws.closed:
#         message = ws.receive()
#         if message is None:  # message is "None" if the client has closed.
#             continue

#         try:
#             response = main(nlp, model, df_data, sent_emb, message)
#         except Exception as err:
#             response = 'Server error'
#             print(err)
#             sys.stdout.flush()
#         clients = ws.handler.server.clients.values()
#         for client in clients:
#             client.ws.send(response)
#             # print('message sent to', client.address)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
