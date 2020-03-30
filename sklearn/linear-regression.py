import pandas as pd 
import numpy as np
import sklearn
import pickle
from sklearn import linear_model


def read_data_from_csv():
    data = pd.read_csv("data/student-math.csv", sep=";")
    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
    return data


def train_model():
    data = read_data_from_csv()

    predict = "G3"

    # data.drop(labal, axis)
    x = np.array(data.drop([predict], 1))
    y = np.array(data[predict])

    highest = 0
    model_main = None
    x_main = None
    y_main = None

    for i in range(10):

        x_train, x_test, y_train, y_test = (
            sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
            )
        # testsize in %

        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)

        acc = model.score(x_test, y_test)

        print("-- Model trained",i,"--" )
        print("Accuracy = ", acc)

        if acc > highest:
            model_main = model
            x_main = x_test
            y_main = y_test
            highest = acc

    return model_main, x_main, y_main


def save_model(model):
    with open("linear_model.pickle", "wb") as f:
        pickle.dump(model, f)

def load_model():
    with open("linear_model.pickle", "rb") as f:
        model = pickle.load(f)

    return model

def save_test_values(x, y):
    # Save the test vaues since the linear model may memorize the rest
    with open("linear-test-values.pickle", "wb") as f:
        pickle.dump((x,y), f)

def load_test_values():
    with open("linear-test-values.pickle", "rb") as f:
        x, y = pickle.load(f)

    return x, y


def main():
    try:
        model = load_model()
        x, y = load_test_values()
        # Test values

    except FileNotFoundError:
        model, x, y = train_model()
        save_model(model)
        save_test_values(x,y)

    print("-- Model Loaded --")
    acc = model.score(x, y)
    print("Model Accuracy:", acc)

    print("Testing manually:")
    predictions = model.predict(x)

    for i in range(10):
        print("-- Predicted grade:", int(round(predictions[i])), "Real grade: ", y[i])

    print("Done")
    print("Model Coeff:", model.coef_)
    print("Model Intercept:", model.intercept_)



main()

    


