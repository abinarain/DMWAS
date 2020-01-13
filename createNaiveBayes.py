######################################################################################################################################################
#CopyRight: Use or Modification of this code requires permission from author.
#Author: Abhishek N. Singh
#Email: abhishek.narain@iitdalumni.com 
#Description: This script creates Naive Bayes model of an encoded Variant file having SNPs and/or DIPs
#It also prints the AUC curve and CrossTab values to check for precision, false positive, true negative, etc. 
#It also generates a file PDValues.csv that has partial dependence of each column sorted in ascending order.
#Usage: python3 programName EncodedFileName.txt PhenotypeFile ColumnNumberwithYvals-1 #Example 5 for 6th column
#Example: python3 createNaiveBayes.py NullMafMultiColDIPsScoredEncoded.txt test.fam 5 
#Note: It would be useful to take the input encoded file which has been already filtered with MAF and Null values and any other discrepancy if any
#Note: the Phenotype file is space delimited and does not have header file name, The EncodedFileName is comma delimited and the header names are the 
#Note: The phenotype column variable should be of 2 values in nature, such as 1 unaffected and 2 affected, 
#and any missing phenotype should be put -9. This is as per PLINK .FAM file input format for phenotype
#column number_ character
#Example EncodedFile:
#-bash-4.2$ head NullMafMultiColDIPsScoredEncoded.txt 
#0_A,0_C,0_G,0_I,0_T,1,2_A,2_C,2_G,2_T,3,4_A,4_C,4_G,4_I,4_T,5,6_A,6_C,6_G,6_T,7,8_A,8_C,8_G,8_T,9,10_A,10_C,10_G,10_I,10_T,11,12_A,12_C,12_G,12_I,12_T,13,14_A,14_C,14_G,14_I,14_T,15,16_A,16_C,16_G,16_I,16_T,17,18_A,18_C,18_G,18_I,18_T,19,20_A,20_C,20_G,20_I,20_T,21,22_A,22_C,22_G,22_T,23,24_A,24_C,24_G,24_T,25,26_A,26_C,26_G,26_I,26_T,27,28_A,28_C,28_G,28_I,28_T,29,30_A,30_C,30_G,30_T,32_A,32_C,32_G,32_T,34_A,34_C,34_G,34_I,34_T,35,36_A,36_C,36_G,36_I,36_T,37,38_A,38_C,38_G,38_I,38_T,39,40_A,40_C,40_G,40_I,40_T,41,42_A,42_C,42_G,42_I,42_T,43,44_A,44_C,44_G,44_I,44_T,45,46_A,46_C,46_G,46_I,46_T,47,48_A,48_C,48_G,48_I,48_T,49,50_A,50_C,50_G,50_I,50_T,51,52_A,52_C,52_G,52_T,53,54_A,54_C,54_G,54_I,54_T,55,56_A,56_C,56_G,56_T,58_A,58_C,58_G,58_I,58_T,59,60_A,60_C#,60_G,60_T,62_A,62_C,62_G,62_I,62_T,63,64_A
#######################################################################################################################################################
import numpy as np
import pandas as pd
import sys
import itertools
import matplotlib.pyplot as plt

class_names = (['Unaffected','Affected'])
#y=np.random.choice([0,1],40) #Assigning 40 values as dependent variable for test purpose
#print(y)
df = pd.read_csv(sys.argv[2], sep =' ', usecols=[int(sys.argv[3])], names=['Y'], header=None) #Read the Phenotype
#print(df.head())
#y = df['Y'].values

data=pd.read_csv(sys.argv[1]) #Reading data genotype encoded
#print(data.head())
dataCombined = pd.concat([data,df], axis=1, sort=False)
#print("Here is a snippet of the combined genotype phenotype data \n")
#print(dataCombined.head())

#Now we drop the rows which have -9 as values, depicting not phenotype values for those members
dataCombined = dataCombined[ dataCombined.Y != -9] #Getting rid of rows which do not have phenotype values present
#dataCombined = dataCombined.drop("-9", axis=0) # Delete all rows with value -9
#print(dataCombined.head())

#Here is our final dependent variables
y = dataCombined['Y'].values #Now we get the y values as numpy arrays
#print(y)
y = y -1 #converting the PLINK format of 1 unaffected 2 affected, to 0 unaffected 1 affected. Comment this out if y is already in 0 1 format
#print(y)
#Here is our final independent variables
data = dataCombined.drop(['Y'], axis=1) #Now we remove the Y since that is dependent variable
#print(data.head())

#Creating Test Train Dataset
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(data,y,test_size=.3,random_state=0)

#Making Logistic Regression Model
from sklearn.naive_bayes import GaussianNB
gr=GaussianNB() #Using Newton-CG solver in the logistic Regression
gr.fit(xtrain,ytrain) #Use the object to create a fit object

#Predicting values for test data
#ypred=lr.predict(data)
ypred=gr.predict(xtest)

#Calculating AUC curve
#from sklearn.metrics import roc_auc_score
#print(roc_auc_score(ytest,ypred))
print("Here is the Confusion Matrix:\n")
print(pd.crosstab(ytest,ypred))

#Here we get the Confusion Matrix
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(ytest, ypred)
np.set_printoptions(precision=2)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Plot Non-Normalized Confusion Matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion Matrix')
#plt.show()
plt.savefig('ConfusionNB.png', bbox_inches='tight')
print("The Confusion Matrix is plotted as ConfusionNB.png")
plt.clf()#clear the figure
plt.cla() #clear the axes

# Plot normalized confusion matrix Uncomment the 3 lines below to plot it.
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized Confusion Matrix')
#plt.show()

#This is another way to get AUC curve
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(ytest, ypred)
roc_auc = metrics.auc(fpr, tpr)
print("ROC AUC value is: "+ str(roc_auc))

#Plot the ROC curve
#import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.show()
plt.savefig('ROCCurveNB.png', bbox_inches='tight')
print("The ROC Curve is plotted as ROCCurveNB.png")
plt.clf() #clear the figure
plt.cla() #clear the axes


#Plotting ROC curve again
#import scikitplot as skplt
#skplt.metrics.plot_roc_curve(ytest, ypred)
#plt.show()

#Getting scores for each column as partial dependence value
from sklearn.inspection import partial_dependence
#b=partial_dependence(lr,features=[0],X=data,percentiles=(0,1))
#print(b)
#print(b[0].max())
#a = np.array(())
listt = [] #Creating a list that will store the maximum dependency value for each column
#listt = ((b[0].max(), 0))
#listt.append([b[0].max(), 0])
#print(data.head(1)) #Here is the head of the data that has column variable names
for i in range(len(data.columns)):
    b=partial_dependence(gr,features=[i],X=data,percentiles=(0,1))
    listt.append([b[0].max(), data.columns.values[i]])
#print(listt)
#Write the listt to a file
#conc = np.vstack(listt)
my_df = pd.DataFrame(listt, columns =['PDValues', 'ColumnName'])
#print(my_df.head())
my_df = my_df.sort_values(by ='PDValues')
#print(my_df.head())
my_df.to_csv('PDValuesNB.csv', index=False)
#a.sort(axis=0)
#np.savetxt('columgSig.csv', listt, delimiter=',')  
#print("Columns Significance Saved to columnSig.csv")

#Now we go for Precision Recall Curve metrics
precision, recall, thresholds = metrics.precision_recall_curve(ytest, ypred)
#Calculate F1 score
f1 = metrics.f1_score(ytest,ypred)
#Calculate the precision-recall AUC
pr_auc = metrics.auc(recall, precision)
#calculate average precision score
ap = metrics.average_precision_score(ytest, ypred)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, pr_auc, ap))
#plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.', label = 'AUPR = %0.2f, f1 = %0.2f, ap = %0.2f' % (pr_auc, f1, ap) )
plt.title('Precision Recall (PR) Curve')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
plt.ylabel('Precision')
plt.xlabel('Recall or Sensitivity')
plt.legend(loc = 'best')
#plt.show()
plt.savefig('PRCurveNB.png', bbox_inches='tight')
print("The Precision Recall curve plotted as PRCurveNB.png")
plt.close()

