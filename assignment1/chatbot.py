import random
import json
import pickle
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()    # helps search the STEM word. Ex: 'work' 'worked' 'working' all stem word is 'work'

intents = json.loads(open('./assignment1/intents.json').read())

words = pickle.load(open('./assignment1/words.pkl', 'rb'))
classes = pickle.load(open('./assignment1/classes.pkl', 'rb'))
model = load_model('./assignment1/chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]= 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25  #if below 25%, we don't use it 
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)  #highest probability first 

    return_list = []

    for r in results: 
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print("Bot is running")

done = False

while not done:
    message = input("")
    if message == "STOP":
        done = True
    else: 
        ints = predict_class(message)
        res = get_response(ints, intents)
        print(res)
        