

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn.linear_model import LogisticRegression


n1=np.genfromtxt('./input/train.csv',delimiter=',')
n2=np.genfromtxt('./input/trainLabels.csv',delimiter=',')
n3=np.genfromtxt('./input/test.csv',delimiter=',')

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
n1= np.r_[n1,n3]
clf.fit(n1,n2)
z= clf.predict(n3)

z1=pd.DataFrame(z,index= np.arange(1,len(z)+1),dtype=int)
z1.to_csv("foo.csv")
