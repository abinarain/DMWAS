#####################################################################################################
#Copyright: Please email the AUTHOR before using this code
#Description: This code replaces the DIPs values in a column variable by the corresponding score for 
#that index at that position.
#Author: Abhishek N. Singh
#Email: abhishek.narain@iitdalumni.com
#Usage$ python3 ReplaceMultiColDIPs.py MultiColFile StepsWhereMultiColsAre ReverseScoreFileName
#Example: python3 ReplaceMultiColDIPs.py multiColumnSplitSample.csv 2 reverse   
#Date: 12th May 2019
######################################################################################################

import numpy as np
import pandas as pd
import sys

inputFileName = "./readwrite/" + sys.argv[1]
reverseFileName = "./readwrite/" + sys.argv[3] 
dataset=pd.read_csv(inputFileName, low_memory=False)
outputFileName = "./readwrite2/" + sys.argv[1] + ".Scored.txt" # Note that the output would be generated in a new directory

for i in range(1,len(dataset.columns), int(sys.argv[2])): #We iterate from column number 1 and move by step of argument given in command line
	scoreFileName = reverseFileName + str(i)
	try:
		scores=pd.read_csv(scoreFileName,sep='\t',header=None) #Reading a TSV file which has the scores of divergence
	except Exception:
		continue #Here we skip to the next value of outer loop and don't get to the inner loop since there could have been exception due to empty file
	for index in scores.index: #Iterate by Index as a nested inner loop in this score file name
		dataset.iloc[scores.iloc[index,0],i] = scores.iloc[index,1] #Here we replaced each of the DIPs in Data for column i 
		# by the Scores Note that the 0th column has the INDEX information and the 1st column has the scores for it

#Now we write the modified dataframe to a file
dataset.to_csv(outputFileName, sep='\t',index=False)
