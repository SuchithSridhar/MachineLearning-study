import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing, model_selection
from time import time

# We're trying to classify the cars into thier class

def main():
    data = pd.read_csv("data/car.data")
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

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)
    
    model = KNeighborsClassifier(n_neighbors=5)
    # n_neighbours is the value of K and needs to be tweaked
    # its the no of neighbours the algorithm considers

    print("-- Training Model --")
    
    start = time()
    model.fit(x_train, y_train)
    end = time()
    
    acc = model.score(x_test, y_test)
    print(f"Train time: {end-start} Accuracy: {acc}")

    print("-- Manual Tests --")
    preds = model.predict(x_test)
    for i in range(10):
        print(f"Prediction :{preds[i]} Actual class:{y_test[i]}")
    print("-- Completed --")

    

main()