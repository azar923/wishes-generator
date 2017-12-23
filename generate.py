from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import json
import argparse
import sys

with open("chars.json") as f:
    chars = json.load(f)

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 100


def build_model(checkpoint_path):
    model = Sequential()
    model.add(LSTM(256, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.load_weights(checkpoint_path)

    return model


def sample(preds, temperature=0.5):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, seed, length=1000, diversity=0.5):

    seed_cut = seed[0:maxlen]
    generated = ''

    print("Generating with seed...")
    print(seed_cut)
    for i in range(length):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(seed_cut):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        seed_cut = seed_cut[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    return generated

model = build_model("model_wishes.hdf5")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help="Beginning of wish to start with (min length 100 symbols)", type=str, default=None)
    parser.add_argument('--length', help="Number of symbols to generate", type=int, default=1000)
    parser.add_argument('--div', help="Diversity (from 0.2 to 1.0)", type=float, default=0.5)

    args, unknown = parser.parse_known_args()

    seed = args.seed
    length = args.length
    diversity = args.div

    if seed is None:
        print("Please specify seed")
        exit()

    if len(seed) < maxlen:
        print("Minimum seed length is 100 symbols")
        exit()

    if diversity < 0.2 or diversity > 1.0:
        print("Please specify diversity in range 0.2 - 1.0")
        exit()

    if length <= 0:
        print("Please specify positive length")
        exit()

    generate_text(model, seed, length, diversity)


