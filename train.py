import tensorflow as tf
import nltk
# bol bhai aa rhi h awaaz
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
import random

words=[]
classes = []
dox = []
avoid = ['.', '?', '!']
json_data = open('dataset.json').read()
dataset = json.loads(json_data)
print(dataset)

for intent in dataset['intents']:
    for pattern in intent['patterns']:

        # tokenizing words
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #adding dox
        dox.append((w, intent['tag']))
 
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in avoid]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
print (len(dox), "dox")
print (len(classes), "classes", classes)
print (len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes)
for doc in dox:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
  
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
    
random.shuffle(training)
training = np.array(training)

train_X = list(training[:,0])
train_y = list(training[:,1])

print("Training data created")

# creating model and building neurons 
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_X[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

# compile model
sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fit model 
histogram = model.fit(np.array(train_X), np.array(train_y), epochs=100, batch_size=5, verbose=1)
model.save('starla.h5', histogram)

print("You can execute main.py now...")
