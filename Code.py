import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#The 'id' feature is not required when making predictions because each id number is unique
df=pd.read_excel("Breast_Cancer_Wisconsin_Dataset.xlsx").drop("id", axis=1)

#View Data
print(df.describe())

#Categorizing the target feature
df["diagnosis"] = df["diagnosis"].astype("category")
#X represents the predictor features
X=df.drop("diagnosis",axis=1)
#y represents the target feature
y=df["diagnosis"]



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Neural Network MLP Classifier
#Random state= 40

from sklearn.neural_network import MLPClassifier as cl
clf = cl(hidden_layer_sizes=(50, ), activation='relu', solver='adam', random_state = 42)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state= 40)
#Use Z score scaler on X_train and X_test
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)
clf = clf.fit(X_train_scaled, y_train)
#y_pred is the predicted values
y_pred = clf.predict(X_test_scaled)

MLP1=round(clf.score (X_test_scaled, y_test),2)
print ('\nAccuracy for MLP classifier at random_state=40 is:', MLP1);
#Displaying histogram plots to compare test values with predicted values
plt.hist(y_test, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for test values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

plt.hist(y_pred, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for predicted values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

#Neural Network MLP Classifier
#Random state= 20

from sklearn.neural_network import MLPClassifier as cl
clf = cl(hidden_layer_sizes=(50, ), activation='relu', solver='adam', random_state = 42)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state= 20)
#Use Z score scaler on X_train and X_test
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)
clf = clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

MLP2=round(clf.score (X_test_scaled, y_test),2)
print ('\nAccuracy for MLP classifier at random_state=20 is:', MLP2);
plt.hist(y_test, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for test values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

plt.hist(y_pred, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for predicted values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

#Neural Network MLP Classifier
#Random state= 1

from sklearn.neural_network import MLPClassifier as cl
clf = cl(hidden_layer_sizes=(50, ), activation='relu', solver='adam', random_state = 42)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state= 1)
#Use Z score scaler on X_train and X_test
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)
clf = clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

MLP3=round(clf.score (X_test_scaled, y_test),2)
print ('\nAccuracy for MLP classifier at random_state=1 is:', MLP3);

plt.hist(y_test, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for test values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

plt.hist(y_pred, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for predicted values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()


#NB
#Random state= 40

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state= 40)
#Use Z score scaler on X_train and X_test
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)
clf = clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

NB1=round(clf.score (X_test_scaled, y_test),2)
print ('\nAccuracy for NB classifier at random_state=40 is:', NB1);
plt.hist(y_test, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for test values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

plt.hist(y_pred, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for predicted values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

#NB
#Random state= 20

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state= 20)
#Use Z score scaler on X_train and X_test
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)
clf = clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

NB2=round(clf.score (X_test_scaled, y_test),2)
print ('\nAccuracy for NB classifier at random_state=20 is:', NB2);
plt.hist(y_test, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for test values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

plt.hist(y_pred, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for predicted values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

#NB
#Random state= 1

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state= 1)
#Use Z score scaler on X_train and X_test
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)
clf = clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

NB3=round(clf.score (X_test_scaled, y_test),2)
print ('\nAccuracy for NB classifier at random_state=1 is:', NB3);

plt.hist(y_test, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for test values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

plt.hist(y_pred, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for predicted values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

#SVC poly
#Random state= 40

from sklearn.svm import SVC #import the tree class
clf = SVC(kernel='poly')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state= 40)
#Use Z score scaler on X_train and X_test
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)
clf = clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

poly1=round(clf.score (X_test_scaled, y_test),2)
print ('\nAccuracy for SVC Polynomial classifier at random_state=40 is:', poly1);
plt.hist(y_test, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for test values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

plt.hist(y_pred, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for predicted values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

#SVC poly
#Random state= 20

clf = SVC(kernel='poly')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state= 20)
#Use Z score scaler on X_train and X_test
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)
clf = clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

poly2=round(clf.score (X_test_scaled, y_test),2)
print ('\nAccuracy for SVC Polynomial classifier at random_state=20 is:', poly2);
plt.hist(y_test, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for test values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

plt.hist(y_pred, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for predicted values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

#SVC poly
#Random state= 1

clf = SVC(kernel='poly')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state= 1)
#Use Z score scaler on X_train and X_test
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)
clf = clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

poly3=round(clf.score (X_test_scaled, y_test),2)
print ('\nAccuracy for SVC Polynomial classifier at random_state=1 is:', poly3);
plt.hist(y_test, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for test values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

plt.hist(y_pred, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for predicted values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

#SVC rbf
#Random state= 40

clf = SVC(kernel='rbf')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state= 40)
#Use Z score scaler on X_train and X_test
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)
clf = clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

rbf1=round(clf.score (X_test_scaled, y_test),2)
print ('\nAccuracy for SVC RBF classifier at random_state=40 is:', rbf1);
plt.hist(y_test, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for test values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

plt.hist(y_pred, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for predicted values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

#SVC rbf
#Random state= 20

clf = SVC(kernel='rbf')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state= 20)
#Use Z score scaler on X_train and X_test
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)
clf = clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

rbf2=round(clf.score (X_test_scaled, y_test),2)
print ('\nAccuracy for SVC RBF classifier at random_state=20 is:', rbf2);
plt.hist(y_test, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for test values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

plt.hist(y_pred, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for predicted values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()


#SVC rbf
#Random state= 1

clf = SVC(kernel='rbf')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state= 1)
#Use Z score scaler on X_train and X_test
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)
clf = clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
rbf3=round(clf.score (X_test_scaled, y_test),2)
print ('\nAccuracy for SVC RBF classifier at random_state=1 is:', rbf3);
plt.hist(y_test, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for test values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

plt.hist(y_pred, bins = 20)
plt.title("Comparison of Number of Observations by Diagnosis for predicted values")
plt.xlabel("Diagnosis (B-Benign, M-Malignant)")
plt.ylabel("Number of Observations")
plt.show()

print("\nAverage accuracy of MLP Classifier is:",round((MLP1+MLP2+MLP3)/3,2))
print("\nAverage accuracy of NB Classifier is:",round((NB1+NB2+NB3)/3,2))
print("\nAverage accuracy of SVC polynomial Classifier is:",round((poly1+poly2+poly3)/3,2))
print("\nAverage accuracy of SVC RBF Classifier is:",round((rbf1+rbf2+rbf3)/3,2))

#The classifiers used, listed from highest average accuracy to lowest, are:
#Support Vector Machine RBF, Neural Network MLP, Support Vector Machine Polynomial, and Naive Bayes.
#In conclusion, 4 Classifiers were able to successfully predict whether a given input would indicate
#a Benign or Malignant tumor. With the most accurate Classifier being the SVM RBF Classifier with 
#an accuracy of 0.98
