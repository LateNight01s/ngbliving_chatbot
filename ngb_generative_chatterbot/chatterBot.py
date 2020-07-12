from chatterbot import ChatBot
from chatterBotTrainers import ChatterBotCorpusTrainer

def trainerFunc(chatbotName):
    trainer = ChatterBotCorpusTrainer(chatbotName)

    trainer.train(
        "chatterbot.corpus.english.conversations"
        )

def inputFunc():
    query=input("Enter you query:>>>").lower()
    return query


chatbot = ChatBot("NGB Living")

#conversation = [
   # "Hello",
   # "Hi there!",
   # "How are you doing?",
   # "I'm doing great.",
   # "That is good to hear",
   # "Thank you.",
   # "You're welcome."
   # "My name is Bantai as I am from Mumbai"
#]

#trainerFunc(chatbot)          #just use it one time for training the model of chatter bot 
print("Chat Bot started.......say bye to exit the conversation")
while True :
    query=inputFunc()
    if 'bye' not in query :
        response = chatbot.get_response(query)
        print(response)
    else:
        print("Bye sir!")
        break
#response = chatbot.get_response(input("Enter you query:>>>"))
#print(response)
