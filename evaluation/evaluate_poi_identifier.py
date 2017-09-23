#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
### Decision Tree Classification
from time import time
from sklearn.tree import DecisionTreeClassifier
t0 = time()
clf = DecisionTreeClassifier()
clf.fit(features, labels)
t1 = time()
print "Training Time:%fs" % (t1-t0)

t0 = time()
pred = clf.predict(features)
t1 = time()
print "Prediction Time:%fs" % (t1-t0)

from sklearn.metrics import accuracy_score
print "Prediction Score:%f" % (accuracy_score(labels, pred))
