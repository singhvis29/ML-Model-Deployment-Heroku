import numpy as np
from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from collections import Counter
import pickle

# Loading Dataset
iris = datasets.load_iris()
X = iris.data#[:, :2]
y = iris.target.reshape(-1,1)

# print(X.shape)
# print(y.shape)

iris_data = np.concatenate((X, y), axis=1)
# print(iris_data.shape)

# Stratified shuffle split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(iris_data, iris_data[:, 4]):
    strat_train_set = iris_data[train_index]
    strat_test_set = iris_data[test_index]

train_X = strat_train_set[:, :4]
train_y = strat_train_set[:, 4]#.reshape(-1, 1)
test_X = strat_test_set[:, :4]
test_y = strat_test_set[:, 4]#.reshape(-1, 1)

# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# Random Forest Classifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train_X, train_y)
print("Feature Importance: {}".format(clf.feature_importances_))

test_pred = clf.predict(test_X)

print("F1 Score on Test Set: {}".format(f1_score(test_y, test_pred, average="weighted")))

# Storing model as a pickle
filename = 'clf_model.pkl'
pickle.dump(clf, open(filename, 'wb'))



