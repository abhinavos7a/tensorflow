import numpy as np
import os.path
import sys


data = open(os.path.join(sys.path[0], 'drake')).read()
vocab = list(set(data))
vocab_len = len(vocab)
n_steps = 25

char_to_id = {ch:id for id,ch in enumerate(vocab)}
id_to_char = {id:ch for id,ch in enumerate(vocab)}

# normalization of weights is one of the key here
Wxh = np.random.randn(vocab_len, 100) * 0.01
Whh = np.random.randn(100,100) * 0.01
Why = np.random.randn(100, vocab_len) * 0.01
bh = np.zeros([1,100])
by = np.zeros([1,vocab_len])
H = np.zeros([1,100])

mxh = np.zeros_like(Wxh)
mhh = np.zeros_like(Whh)
mhy = np.zeros_like(Why)
mbh = np.zeros_like(bh)
mby = np.zeros_like(by)

def get_next_io(data, itr):
    start = itr * n_steps
    end = (itr+1) * n_steps

    input = data[start:end]
    output = data[start+1:end+1]

    return input, output

def one_hot_encodings(chars):
    char_encodings = []
    char_indexes = []
    for index, char_value in enumerate(chars):
        x = np.zeros((1,vocab_len))
        x[0][char_to_id[char_value]] = 1
        char_encodings.append(x)
        char_indexes.append(char_to_id[char_value])

    return char_encodings, char_indexes

def print_sample(init_x, init_h=np.zeros([1,100]), n = 200):
    output = []

    for i in range(0, n):
        if i == 0:
            h = init_h
            x = init_x

        a = np.dot(x, Wxh) + np.dot(h, Whh) + bh
        h = np.tanh(a)
        z = np.dot(h, Why) + by
        y = np.exp(z)/np.sum(np.exp(z)) # probability distribution

        x_id = np.random.choice(a = range(vocab_len), p = y.ravel())
        #x_id = np.argmax(y)
        output.append(id_to_char[x_id])
        x = np.zeros((1,vocab_len))
        x[0][x_id] = 1

    print('\n'+''.join(output) + '\n')
    return output

def forward_pass(X, init_h = np.zeros([1,100])):
    H,Yhat = [],[]
    H.append(np.reshape(init_h, (1,100))) # initial h
    for index, x in enumerate(X):
        a = np.dot(x, Wxh) + np.dot(H[index],Whh) + bh
        h = np.tanh(a)
        z = np.dot(h, Why) + by
        y = np.exp(z)/np.sum(np.exp(z))

        Yhat.append(y)
        H.append(h)

    return Yhat, H

def backward_prop(Yhat, H, Y, X, Y_original):

    dwhy = np.zeros_like(Why)
    dwxh = np.zeros_like(Wxh)
    dwhh = np.zeros_like(Whh)
    dby = np.zeros_like(by)
    dbh = np.zeros_like(bh)
    dh_previous = np.zeros([1,100])
    loss = 0

    for i in range(0, n_steps):
        ith_step = n_steps - i - 1

        # loss computation

        yhat = Yhat[ith_step]
        y_original = Y_original[ith_step]
        x = X[ith_step]
        y = Y[ith_step]
        h = H[ith_step + 1] # H has size n+1
        h_previous = H[ith_step]

        loss += -np.log(yhat[0][y_original])
        dwhy += np.dot(np.transpose(h), yhat - y)
        dby += yhat - y

        dh = (np.dot(yhat - y, np.transpose(Why))) + dh_previous

        dwxh += np.dot(np.transpose(x), dh * (1- h*h))
        dwhh += np.dot(np.transpose(h_previous), dh * (1- h*h)) # problematic

        dbh += dh * (1- h*h)

        dh_previous = np.dot(dh * (1- h*h), np.transpose(Whh))

    return dwhy, dwxh, dwhh, dby, dbh, loss


iteration = 0
epochs = 0
n_iteration = 0
alpha = 1e-1 # 0.1
smooth_loss = -np.log(1.0 / vocab_len) * n_steps  # loss at iteration 0

while True:
    n_iteration += 1
    if (iteration + 2) * n_steps > len(data):
        epochs += 1
        iteration = 0
    input, output = get_next_io(data, iteration)
    X, X_original = one_hot_encodings(input)
    Y, Y_original = one_hot_encodings(output)
    Yhat, H = forward_pass(X, H[-1])
    dwhy, dwxh, dwhh, dby, dbh, loss = backward_prop(Yhat, H, Y, X, Y_original)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

# adagrad optimizer
    mhy += dwhy * dwhy
    mxh += dwxh * dwxh
    mhh += dwhh * dwhh
    mby += dby * dby
    mbh += dbh * dbh

    Why = Why - alpha * dwhy / np.sqrt(mhy + 1e-8)
    Wxh = Wxh - alpha * dwxh / np.sqrt(mxh + 1e-8)
    Whh = Whh - alpha * dwhh / np.sqrt(mhh+ 1e-8)
    by = by - alpha * dby / np.sqrt(mby+ 1e-8)
    bh = bh - alpha * dbh / np.sqrt(mbh+ 1e-8)

    if n_iteration % 100 == 0:
        print_sample(X[0], init_h=H[-1])
        print('epoch: {} iteration: {} loss: {}'.format(epochs, n_iteration, loss))
    iteration += 1

