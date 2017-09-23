#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  
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