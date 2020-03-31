from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pickle


def save_model(model):
    with open("NeuralModel.pickle", "wb") as f:
        pickle.dump(model, f)

def load_model():
    with open("NeuralModel.pickle", "wb") as f:
        model =pickle.load(f)
    return model

def showImage(image_data):
    plt.imshow(image_data, cmap=plt.cm.binary)
    plt.show()


data = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = data.load_data()

# labels 0-9

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# showImage(train_images[0])
# Display one of the images

train_images = train_images / 255
test_images = test_images / 255

# 28x28 with values 0-1

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128, activation="relu")) # Rectifier linear unit
model.add(keras.layers.Dense(10, activation="softmax")) # percentage output - probability

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)
# epochs - the number of times it sees the same image

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"accuracy = {test_acc}")
print(f"Testloss = {test_loss}")

# save_model(model)

prediction = model.predict(test_images)

for i in range(10):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(f"Actual Object : {class_names[test_labels[i]]}")
    plt.title(f"Prediction: {class_names[np.argmax(prediction[i])]}")
    plt.show()



