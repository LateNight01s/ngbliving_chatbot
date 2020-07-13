import flask
from ngb_chatbot import main

def chatbot(request):
    if request.method == 'GET':
        return f'Get request', 200

    req = request.get_json(silent=True)
    if req and 'data' in req:
        message = req['data']['message']
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
