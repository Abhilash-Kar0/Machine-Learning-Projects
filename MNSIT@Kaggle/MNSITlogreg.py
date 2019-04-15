
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import os
# #print(os.listdir("../input"))
# df = pd.read_csv( './input/train.csv')
# df1 =pd.read_csv( './input/trainLabels.csv')
# df2 = pd.read_csv( './input/test.csv')
# #print(df.values)
# # Any results you write to the current directory are saved as output.

 n1=np.genfromtxt('./input/train.csv',delimiter=',')
 n2=np.genfromtxt('./input/trainLabels.csv',delimiter=',')
 n3=np.genfromtxt('./input/test.csv',delimiter=',')
 poly = sklearn.preprocessing.PolynomialFeatures(degree=2)
 n1=poly.fit_transform(n1)
 Logreg = LogisticRegression(C=1000, solver='lbfgs', multi_class='ovr')
 Logreg.fit(n1,n2)
 n3=poly.fit_transform(n3)
 z= Logreg.predict(n3)
z1=pd.DataFrame(z,index= np.arange(1,len(z)+1),dtype=int)
# #print(z1)
# #z1.columns=["Id","Solution"]
z1.to_csv("foo.csv")
# #np.savetxt("foo.csv",z,delimiter="\n")

