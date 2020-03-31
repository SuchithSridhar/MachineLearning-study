from tensorflow import keras
import numpy as np

data = keras.datasets.imdb


word_index = data.get_word_index()
word_index = {k: (v+3) for k, v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for key, value in word_index.items()])
# word --> int
# int --> word


def decode(text):
    return ' '.join([reverse_word_index.get(i, "?") for i in text])


def encode(text):
    unk = word_index.get("<UNK>")
    return [1] + [word_index.get(i, unk) for i in text]


def train_save_model():
    ((train_data, train_labels),
     (test_data, test_labels)) = data.load_data(num_words=88_000)

    train_data = keras.preprocessing.sequence.pad_sequences(
        train_data,
        value=word_index["<PAD>"],
        padding="post",
        maxlen=250
    )

    test_data = keras.preprocessing.sequence.pad_sequences(
        train_data,
        value=word_index["<PAD>"],
        padding="post",
        maxlen=250
    )

    model = keras.Sequential()

    # Word vectors, good and great are similar words
    model.add(keras.layers.Embedding(88_000, 16))

    # puts the above 16D into 1D
    model.add(keras.layers.GlobalAveragePooling1D())

    # 16 neurons choosen for an arbitrary reason
    model.add(keras.layers.Dense(16, activation="relu"))

    # Sigmoid ---> output wtihin 0 - 1
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    # validation data
    x_val = train_data[:10_000]
    x_train = train_data[10_000:]

    y_val = train_labels[:10_000]
    y_train = train_labels[10_000:]

    fitModel = model.fit(x_train, y_train, epochs=40,
                         batch_size=512,
                         validation_data=(x_val, y_val),
                         verbose=1)

    # batchsize - how many reviews each cycle since too many reviews present

    results = model.evaluate(test_data, test_labels)

    print(f"Results: {results}")
    print()
    model.save("model.h5")
    print("-- Saved Model --")

    print()
    test_review = test_data[1]
    predict = model.predict([test_review])[0]
    print("Review: ")
    print(decode(test_review))
    print(f"Prediction : {predict}")
    print(f"Actuall : {test_labels[1]}")


def test_on_real_data():
    model = keras.models.load_model("model.h5")
    with open("test_review.txt") as f:
        string = f.read()

    for item in '''("',.:\\n!-?)''':
        string = string.replace(item, " ")

    while "  " in string:
        string = string.replace("  ", " ")
        # replace 2 spaces with one space

    data = string.split()

    data = encode(data)

    data = keras.preprocessing.sequence.pad_sequences(
        [data],
        value=word_index["<PAD>"],
        padding="post",
        maxlen=250
    )

    # print(data)

    prediction = model.predict([data])[0]
    print("Prediction: ", prediction)


test_on_real_data()
