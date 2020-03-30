import sklearn
from sklearn import metrics
from sklearn import model_selection
from sklearn import datasets
from sklearn import svm

cancer_data = datasets.load_breast_cancer()

print("Cancer Features: ", cancer_data.feature_names)
print("Cancer Targets: ", cancer_data.target_names)

x = cancer_data.data
y = cancer_data.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.1)


print("-- Training classifier without kernel --")
clf = svm.SVC()

clf.fit(x_train, y_train)
predicted = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, predicted)

print(f"Accuracy = {acc}")
