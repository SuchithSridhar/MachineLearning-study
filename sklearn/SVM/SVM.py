import sklearn
from sklearn import metrics
from sklearn import model_selection
from sklearn import datasets
from sklearn import svm
import pickle

data = datasets.load_breast_cancer()

print("Cancer Features: ", data['feature_names'])
print("Cancer Targets: ", data['target_names'])

x = data["data"]
y = data["target"]



def save_model(model, x_test, y_test):
    with open("SVM-Model.pickle", "wb") as f:
        pickle.dump(model, f)

    with open("SVM-Test-Values.pickle", "wb") as f:
        pickle.dump((x_test, y_test), f)


def load_model():
    with open("SVM-Model.pickle", "rb") as f:
        model = pickle.load(f)

    with open("SVM-Test-Values.pickle", "rb") as f:
        x, y = pickle.load(f)

    return model, x, y

print("-- Training classifier with default kernel --")


x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.1)
clf = svm.SVC()

clf.fit(x_train, y_train)
predicted = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, predicted)

print(f"Accuracy = {acc}")

print()
print("-- Training with linear kernel --")
clf = svm.SVC(kernel="linear")

clf.fit(x_train, y_train)
predicted = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, predicted)

print(f"Accuracy = {acc}")