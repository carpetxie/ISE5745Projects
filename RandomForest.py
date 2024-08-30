from re import X
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()
#Plot the first four digits


for i in range(4):
  plt.matshow(digits.images[i])
  plt.show

#Preprocessing
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Model
clf = RandomForestClassifier(max_depth=15, random_state=0, n_estimators = 20)
clf.fit(X_train, y_train)

#Evaluation
correct = 0
total = len(X_test)

predictions = clf.predict(X_test)

for pred, true_labels in zip(predictions, y_test):
  if pred == true_labels:
    correct +=1

print(correct / total)

from re import X
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

digits = load_digits()
#Preprocessing
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Model
clf = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=(50, 20, 50), activation = "identity", solver = "adam")
clf.fit(X_train, y_train)

#Evaluation
correct = 0
total = len(X_test)

predictions = clf.predict(X_test)

for pred, true_labels in zip(predictions, y_test):
  if pred == true_labels:
    correct +=1

print(correct / total)

#97.77%

from re import X
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

digits = load_digits()

#Preprocessing
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
'''
#Top 5 Features
clf = RandomForestClassifier(max_depth=15, random_state=0, n_estimators = 20)
selector = RFE(clf, n_features_to_select=5, step=1)
selector = selector.fit(X_train, y_train)
selector.ranking_
#Index Location of Top 5 Features
rankindex = []
for index, item in enumerate(selector.ranking_):
  if item == 1:
    rankindex.append(index)
print(rankindex)
'''
#Model
clf = RandomForestClassifier(max_depth=15, random_state=0, n_estimators = 20)
selector = RFE(clf, n_features_to_select=25, step=1)
selector = selector.fit(X_train, y_train)

#Evaluation
correct = 0
total = len(X_test)

predictions = selector.predict(X_test)

for pred, true_labels in zip(predictions, y_test):
  if pred == true_labels:
    correct +=1

print(correct / total)

#96.8%

from re import X
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

digits = load_digits()

#Preprocessing
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Model
clf = RandomForestClassifier(max_depth=15, random_state=0, n_estimators = 20)
selector = RFE(clf, n_features_to_select=8, step=1)
selector = selector.fit(X_train, y_train)

#Evaluation
correct = 0
total = len(X_test)

predictions = selector.predict(X_test)

for pred, true_labels in zip(predictions, y_test):
  if pred == true_labels:
    correct +=1

print(correct / total)

#91.11%
