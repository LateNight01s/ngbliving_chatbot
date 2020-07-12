from chatterbot import ChatBot
import os
import sys
import csv
import time
import chatterbot
import chatterbot_corpus
from multiprocessing import Pool, Manager
from dateutil import parser as date_parser
from chatterbot.conversation import Statement
from chatterbot.tagging import PosHypernymTagger
from chatterbot import utils

class Trainer(object):
    #Base class for all other trainer classes.

    def __init__(self, chatbot, **kwargs):
        self.chatbot = chatbot

        environment_default = os.getenv('CHATTERBOT_SHOW_TRAINING_PROGRESS', True)
        self.show_training_progress = kwargs.get(
            'show_training_progress',
            environment_default
        )

    def get_preprocessed_statement(self, input_statement):
               
        for preprocessor in self.chatbot.preprocessors:
            input_statement = preprocessor(input_statement)

        return input_statement

    def train(self, *args, **kwargs):

        raise self.TrainerInitializationException()

    class TrainerInitializationException(Exception):

        def __init__(self, message=None):
            default = (
                'A training class must be specified before calling train(). '
                'See http://chatterbot.readthedocs.io/en/stable/training.html'
            )
            super().__init__(message or default)

    def _generate_export_data(self):
        result = []
        for statement in self.chatbot.storage.filter():
            if statement.in_response_to:
                result.append([statement.in_response_to, statement.text])

        return result

    def export_for_training(self, file_path='./export.json'):
        import json
        export = {'conversations': self._generate_export_data()}
        with open(file_path, 'w+') as jsonfile:
            json.dump(export, jsonfile, ensure_ascii=False)


class ChatterBotCorpusTrainer(Trainer):
    def train(self, *corpus_paths):
        from chatterbot.corpus import load_corpus, list_corpus_files

        data_file_paths = []

        # Get the paths to each file the bot will be trained with
        for corpus_path in corpus_paths:
            data_file_paths.extend(list_corpus_files(corpus_path))

        for corpus, categories, file_path in load_corpus(*data_file_paths):

            statements_to_create = []

            # Train the chat bot with each statement and response pair
            for conversation_count, conversation in enumerate(corpus):

                if self.show_training_progress:
                    utils.print_progress_bar(
                        'Training ' + str(os.path.basename(file_path)),
                        conversation_count + 1,
                        len(corpus)
                    )

                previous_statement_text = None
                previous_statement_search_text = ''

                for text in conversation:

                    statement_search_text = self.chatbot.storage.tagger.get_bigram_pair_string(text)

                    statement = Statement(
                        text=text,
                        search_text=statement_search_text,
                        in_response_to=previous_statement_text,
                        search_in_response_to=previous_statement_search_text,
                        conversation='training'
                    )

                    statement.add_tags(*categories)

                    statement = self.get_preprocessed_statement(statement)

                    previous_statement_text = statement.text
                    previous_statement_search_text = statement_search_text

                    statements_to_create.append(statement)

            self.chatbot.storage.create_many(statements_to_create)


def inputFunc():
    query=input("Enter you query:>>>").lower()
    return query
    
def trainerFunc(chatbotName):
    trainer = ChatterBotCorpusTrainer(chatbotName)
    trainer.train(
        "chatterbot.corpus.english.conversations"
        )
    return   

chatbot = ChatBot("Bantai")

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
