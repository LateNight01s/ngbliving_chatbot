from chatterBotTrainers import ChatterBotCorpusTrainer

def trainerFunc(chatbotName):
    trainer = ChatterBotCorpusTrainer(chatbotName)

    trainer.train(
        "chatterbot.corpus.english.conversations"
        )