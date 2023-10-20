import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()    # helps search the STEM word. Ex: 'work' 'worked' 'working' all stem word is 'work'

intents = json.loads(open('./assignment1/intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)  #tokenize splits the text up into a collection of single words
        words.extend(wordList)      #extend: takes content and appending to list
        documents.append((wordList, intent['tag']))     #the words in the word list BELONGS to the tag/category
        if intent['tag'] not in classes:    #check if the tag already exists, if not, add it 
            classes.append(intent['tag'])

print(documents)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('./chatbot/words.pkl', 'wb'))
pickle.dump(classes, open('./chatbot/classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

# neural networks: bag of words - setting individual word values to 0 or 1 depending if it occurs in the pattern
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        #check if the word occurs in the pattern, 1 = occurs, 0 = doesn't occur
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

# Load your data and preprocess it as needed
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Use mean squared error (MSE) as the loss
mse_loss = tf.keras.losses.MeanSquaredError()

# Create a normal gradient descent optimizer with learning rate
custom_lr = 0.01
optimizer = tf.keras.optimizers.SGD(learning_rate=custom_lr, momentum=0.0, nesterov=False)

# Compile the model with the MSE loss and gradient descent optimizer
model.compile(loss=mse_loss, optimizer=optimizer, metrics=['mean_squared_error'])

# Train your model with gradient descent
model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)
model.save('./chatbot/chatbot_model.h5')
print('Done')
