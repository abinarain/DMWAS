#CopyRight: Please take permission before using this script. Most importantly, please cite this work if you use this script.
#
#Citation: Abhishek N. Singh, DMLWAS: Deep & Machine Learning Wide Association Studies with ExhaustiveDNN such as for genome variations linked to phenotype or drug repositioning
#
#++++++++++++++++ Author: Abhishek N. Singh Email: abhishek.narain@iitdalumni.com Date: 12th January 2020 Purpose: Does and Exhaustive Neural Network model building for a range of hidden layers and hidden unit values. Example: Does an Exhaustive Deep Neural Network execution on encoded data for genotype and the corresponding phenotype values. However, it can be used for any other purpose too.
################################################################

import numpy as np
import pandas as pd
import os
#importing basic library for preprocessing

data=pd.read_csv("MultiColDIPsScoredEncoded.txt") #reading data
x= data.values#converting into array
y=pd.read_csv("Phenotypes.txt") #Here we get the Y phenotype values
y=y.values[:,1]
c=[]
for i in data:
    if data[i].isnull().any():
        c.append(i)
#getting to list of column for nulll values

c.reverse() 
for i in c:
    data=data.drop(i,axis=1)
#dropping null columns from back direction in order to prevent column number change
x=data.values
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.3,random_state=0)
classifier = Sequential()

def add_layer(i,clf):
    clf.add(Dense(output_dim = i, 
                         init = 'uniform', 
                         activation = 'relu'))
    return(clf)
    
#module for adding layer after initialization i will be number of hidden units clf will be model constructed  

def initiate(clf,column_no,i):
    clf.add(Dense(output_dim = i,
                  init = 'uniform',
                  activation = 'relu',
                  input_dim = column_no))
    return(clf)
#we are initiating our neural with input layer number and first hidden layer hidden units number
    
def output(clf):
    clf.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    return(clf)
#here we are getting output from neural network

def compiler(clf):
    clf.compile(optimizer = 'adam',
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])
    return(clf)
#here we are compiling our model

earlyStopping=keras.callbacks.EarlyStopping(monitor="val_loss",patience=200,verbose=1,mode="auto")
#we are creating model for early stop

def fitMyModel(clf,x,y,x1,y1,b=10000,n=1000):
    clf.fit(x, y, batch_size = b, nb_epoch = n,callbacks=[earlyStopping],validation_data=[x1,y1])
    return(clf)
#n=1000 is the epoch default
#this module is used for fitting
#b is batch size by default we are keeping it 10000 as number of roows to get

from sklearn.model_selection import StratifiedKFold
#for cross validation

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)

def savemodel(k,i,clf):
    model_json = clf.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    clf.save_weights("model/model.h5")
    print("Saved model to disk")
#here we are saving model skeleton and its weights and bias in model directory

def deletePrevModel():
    ls=os.listdir("model")
    for i in ls:
        os.remove("model/{}".format(i))
#when we will get new best values we have to delete old model and weights

m=2
n=8
#m is min  number of hidden layer
#n is max number of hidden layer

p=8 #min number of hidden units
q=12 #max number of hidden units
b=10 # is batch size
n=1000 #epoch

from sklearn.metrics import confusion_matrix

best_score=0
for k in range(p,q,1):  # loop for hidden unit 
    for i in range(m,n,1): # loop for hidden  layer
        clf = Sequential()
        clf=initiate(clf,x.shape[1],k)# here we are initiating model k in number of first hidden units
        for j in range(i): #executing each hidden layer
            clf=add_layer(k,clf)
        clf =output(clf)
        clf=compiler(clf)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
        scores=[]
        for train, test in kfold.split(x, y):
            clf=fitMyModel(clf,x[train],y[train],x[test],y[test],b,n)
            score = clf.evaluate(x[test], y[test], verbose=0)
            scores.append(score[1])
        avg_score=np.mean(scores)
        if avg_score>best_score:
            deletePrevModel()
            savemodel(k,i,clf)
            best_score=avg_score
	    # Predicting the Test set results
            y_pred = clf.predict(xtest)
            y_pred = (y_pred > 0.5)
            cm = confusion_matrix(ytest, y_pred)
            #writing to file
            f=open("model/score.txt","w") 
            f.write("hidden units: {} \nhidden layers: {} \nbest_score:{} \nconfusion_matrix:{}".format(k,i,best_score,cm))
            f.close()
                
        del(clf)
#here either early stopping condition is met or epoch end is met the model will save and terminate for each value in loop    
        
