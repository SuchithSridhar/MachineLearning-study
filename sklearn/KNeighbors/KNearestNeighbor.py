import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing, model_selection
from time import time
import pickle

# We're trying to classify the cars into thier class

DATAFILE = "car.data"

def save_model(model, x_test, y_test):
    with open("KNN_model.pickle", 'wb') as f:
        pickle.dump(model, f)

    with open("KNN_test-values", "wb") as f:
        pickle.dump((x_test, y_test), f)


def read_model():
    with open("KNN_model.pickle", 'rb') as f:
        model = pickle.read(f)

    with open("KNN_test-values", "rb") as f:
        x, y = pickle.read(f)

    return model, x, y


def main():
    data = pd.read_csv(DATAFILE)
    preObj = preprocessing.LabelEncoder()

    items = []
    names = "buying,maint,doors,persons,lug_boot,safety,class".split(",")
    for name in names:
        items.append(preObj.fit_transform(list(data[name])))
        # This is to convert all the named attributes into
        # integer attribs so that its easier to process
    
    

    # zip will create tuples, where each tuple is a row
    x = list(zip(*items[:-1]))
    y = list(items[-1]) # only the class field

    
    

    print("-- Training Model --")

    saved_model = None
    saved_acc = 0
    saved_k = 0
    saved_test = ()
    for j in range(25):
        print(f" -- Round {j+1} --")
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)
        for i in range(1,5):
            k = (i*2)+1

            model = KNeighborsClassifier(n_neighbors=k)
            # n_neighbours is the value of K and needs to be tweaked
            # its the no of neighbours the algorithm considers

            start = time()
            model.fit(x_train, y_train)
            end = time()
        
            acc = model.score(x_test, y_test)
            print(f"Kvalue: {k} Train time: {end-start} Accuracy: {acc}")
            if acc>saved_acc:
                saved_k = k
                saved_acc = acc
                saved_model=model
                saved_test = (x_test, y_test)

    model = saved_model
    # save_model(model, *saved_test)

    print(f"-- Best Model for k = {saved_k} Accuracy = {saved_acc}--")
    print("-- Manual Tests --")
    preds = model.predict(x_test)
    for i in range(10):
        print(f"Prediction :{preds[i]} Actual class:{y_test[i]}")
    print("-- Completed --")

    

main()