import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import roc_curve
from sklearn.metrics import zero_one_loss
from sklearn.neural_network import MLPClassifier
import math 



data = pd.read_csv("H:\\Data sets\\HR_comma_sep.csv")



#print(data.head)
#print(data.info())
#print(data.isnull().sum()) 
#print(data.shape)



max_threshold = data["satisfaction_level"].quantile(0.999)
min_threshold = data["satisfaction_level"].quantile(0.001)
data = data[(data["satisfaction_level"] < max_threshold) & (data["satisfaction_level"] >
            min_threshold)] 
#print("satisfaction_leve: ",data.shape)

max_threshold = data["last_evaluation"].quantile(0.999)
min_threshold = data["last_evaluation"].quantile(0.001)
data = data[(data["last_evaluation"] < max_threshold) & (data["last_evaluation"] >
            min_threshold)] 
#print("last_evaluation: ",data.shape)


max_threshold = data["average_montly_hours"].quantile(0.999)
min_threshold = data["average_montly_hours"].quantile(0.001)
data = data[(data["average_montly_hours"] < max_threshold) & (data["average_montly_hours"] >
            min_threshold)] 
#print("average_montly_hours: ",data.shape)

print("Data shape is :",data.shape)

dum = pd.get_dummies(data.Department)
data = pd.concat([data,dum],axis="columns")
data = data.drop(["Department"],axis="columns")
data = data.drop(["RandD"],axis="columns")


dum = pd.get_dummies(data.salary)
data = pd.concat([data,dum],axis="columns")
data = data.drop(["salary"],axis="columns")
data = data.drop(["high"],axis="columns")
#print(data.shape)

target = data["left"]
data = data.drop(["left"] ,axis="columns")
print("X shape : ",data.shape)
print("Y shape :",target.shape)


x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2)
print("The length of traning data is: ",len(x_train))
#print("The shape of traning data is: ",x_train.shape)
print("The length of testing data is: ",len(x_test))
print("__________________________________________________")


model = SVC(C=20,kernel="rbf",degree=3,coef0=1.5,gamma="auto")
model.fit(x_train,y_train)
print("The accuracy of SVC is :",math.ceil(100*model.score(x_test,y_test)),"%")


cm = confusion_matrix(y_test,model.predict(x_test))
plt.figure(figsize=(7,5))
sn.heatmap(cm,annot=True)
plt.xlabel("predict")
plt.ylabel("truth")

fpr,tpr,threshold = roc_curve(y_test,model.predict(x_test))
print("Fpr is :",fpr,"\n tpr is :",tpr,"\n threshold is :",threshold)

print("The loss curve is :",zero_one_loss(y_test,model.predict(x_test),normalize=False))
print("_____________________________________________________")


model2 = MLPClassifier(activation="tanh",solver="adam",learning_rate="constant",
                       alpha=0.001,hidden_layer_sizes=(100,30),random_state=10,
                       batch_size="auto",max_iter=3000)
model2.fit(x_train,y_train)
print("the accuracy of ANN is : ",math.ceil(100*model2.score(x_test,y_test)),"%")


cm2 = confusion_matrix(y_test,model2.predict(x_test))
plt.figure(figsize=(7,5))
sn.heatmap(cm2,annot=True)
plt.xlabel("predict")
plt.ylabel("truth")


fpr2,tpr2,threshold2 = roc_curve(y_test,model2.predict(x_test))
print("Fpr is :",fpr2,"\n tpr is :",tpr2,"\n threshold is :",threshold2)

print("The loss curve is :",zero_one_loss(y_test,model2.predict(x_test),normalize=False))
