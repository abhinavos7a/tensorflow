from tensorflow import keras
import tensorflow as tf
import numpy as np
import random as rand
import os.path
import sys


text = open(os.path.join(sys.path[0], 'drake')).read()
vocab = list(set(text))
vocab_len = len(vocab)
time_steps = 25
n_epochs = 50

char_to_id = {ch:id for id,ch in enumerate(vocab)}
id_to_char = {id:ch for id,ch in enumerate(vocab)}

int_data = [char_to_id[ch] for ch in text]

def get_next_io(data, itr):
    start = itr * time_steps
    end = (itr+1) * time_steps

    input = data[start:end]
    output = data[start+1:end+1]

    return input, output

def one_hot_encodings(char):
    char_encoding = np.zeros(vocab_len)
    char_encoding[char_to_id[char]] = 1

    return char_encoding


def get_next_io(data, itr):
    start = itr * time_steps
    end = (itr+1) * time_steps

    input = data[start:end]
    output = data[start+1:end+1]

    return input, output

def generate_X_Y(text):
    iteration = 0
    X = []
    Y = []
    while (iteration + 2) * time_steps < len(text):
        x,y=[],[]
        input, output = get_next_io(text, iteration)
        for ch in input:
            x.append(one_hot_encodings(ch))

        y = one_hot_encodings(output[-1])

        X.append(x)
        Y.append(y)
        iteration += 1

    return X, Y

#build the model
model = keras.Sequential()
rnn = keras.layers.SimpleRNN(units=64,
                             activation='tanh',
                             use_bias=True,
                             kernel_initializer='glorot_uniform',
                             recurrent_initializer='orthogonal',
                             bias_initializer='zeros',
                             kernel_regularizer=None,
                             recurrent_regularizer=None,
                             bias_regularizer=None,
                             activity_regularizer=None,
                             kernel_constraint=None,
                             recurrent_constraint=None,
                             bias_constraint=None,
                             dropout=0.0,
                             recurrent_dropout=0.0,
                             return_sequences=False,  # yhat only at last step
                             return_state=False,  # return the last h
                             go_backwards=False,
                             # stateful=True,# use last h as input to next batch
                             unroll=True,
                             input_shape=(time_steps, vocab_len))
model.add(rnn)
model.add(keras.layers.Dense(units=vocab_len, activation='softmax'))

#compile the model
model.compile(optimizer=tf.train.AdagradOptimizer(learning_rate=1e-1),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
print (model.summary())

X,Y = generate_X_Y(text)

# X is list of time_steps, vocab_len
history = model.fit(x=np.array(X), y=np.array(Y), epochs=n_epochs)
print(history)
print(model.summary())

def gen_seq(seed_chars, seq_len):
    output_chars = seed_chars
    for i in range(seq_len):
        output_chars = output_chars[-seq_len:]
        yhat = model.predict(np.array([[one_hot_encodings(ch) for ch in output_chars]]))
        # yhat (25,75)
        last_yhat = yhat[0]
        x_id = np.random.choice(a = range(vocab_len), p = last_yhat.ravel())
        new_char = id_to_char[x_id]
        output_chars += new_char

    print(output_chars)

while True:
    rand_index = rand.randint(0, len(text) - time_steps)
    gen_seq(text[rand_index: rand_index+time_steps], time_steps)
