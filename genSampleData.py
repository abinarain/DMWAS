############################################################################################################################################
#Copyright: Please contact the author by email before using this script or modifying it
#Description: This script generates variation such as SNPs and DIPs randomly
#Author: Abhishek N. Singh
#Email: abhishek.narain@iitdalumni.com
#Usage$ python3 genSampleData.py <no. of genotypes columns> <no. of individual rows> <frequency of one of how many> <max size of insertions> <outputFileName>
#############################################################################################################################################
import numpy as np
import pandas as pd
import sys
b=["A","T","G","C"] #From here the choice of SNPs will be picked up

#########################################Functions Below##############################################################

def DIPsGen(maxDIP): #This is the function to generate DIPs
    b=["A","T","G","C"] #The nucleotides that go into the Insertion sequence
    x=""
    c=np.random.randint(2,maxDIP) #range from size of 2 to maximum length provided, say 30
    for i in range(c):
        y=np.random.choice(b)
        x=x+y
    return(x)

def ColVar(rows, oneInNumber, maxDIP): #Here we get variation data with as much rows as possible and the frequency of one in a number say 10
    data1=np.array(())
    for i in range(rows):
        ex=np.random.randint(oneInNumber)
        if  not ex==6: #This makes sure than only 1 in the number of cases would insert a DIP. 6 is chosen as an arbitrary number between 1 and 10
            alph=np.random.choice(b)
            data1=np.append(data1,alph)
        else:#Now one case has been that of DIPs insertion
            alph=DIPsGen(maxDIP)
            data1=np.append(data1,alph)
    return(data1)

def MulColVar(cols, rows, oneInNumber, maxDIP): #This function adds as many columns as you pass the value by cols
    dataWithFrame=pd.DataFrame()
    #dataColumns = np.array(())
    for i in range(cols):
        #print(i)    
        data = ColVar(rows, oneInNumber, maxDIP) #return of data to get the values each time new data of varied DIPs and SNPs
        #print(data) #remove hash to see the data list
        #dataColumns = np.concatenate(dataColumns,data,0)
        dataWithFrame[i] = data
    return(dataWithFrame)
##########################################Execution Below################################################################
df = MulColVar(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])) # column phenotypes, rows of individuals, frequency of one out of how many number, insertion max size
df.to_csv(sys.argv[5],header=True,index=False)
