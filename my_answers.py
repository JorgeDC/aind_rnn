import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    series_size = np.shape(series)[0]
    for i in range(0, series_size-window_size):
        batch = series[i:i+window_size]
        X.append(batch)
        y.append(series[i+window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    lstm_layer = LSTM(5, input_shape=(window_size,1))
    model.add(lstm_layer)
    dense_layer = Dense(1)
    model.add(dense_layer)
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    for char_to_replace in punctuation:
        print(char_to_replace)
        text = text.replace(char_to_replace,' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    X = []
    y = []

    series_size = len(text)

    for i in range(0, series_size-window_size, step_size):
        batch = text[i:i+window_size]
        X.append(batch)
        y.append(text[i+window_size])


    return X,y

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    lstm_layer = LSTM(200, input_shape=(window_size, 52))
    model.add(lstm_layer)
    dense_layer = Dense(num_chars, activation='softmax')
    model.add(dense_layer)
    return model
