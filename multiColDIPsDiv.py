###############################################################################################################################################
#Copyright: Please contact the author before using this tool. You can write to him at his email
#Email: abhishek.narain@iitdalumni.com
#Purpose: Multi Column DIPs extraction and Divergence calculation
#Domain: Bioinformatics
#Dependencies: T-Coffee, External Data for DIPS such as that below:
#0 T 0.0
#1 I AATTGGC
#Note: The DIPs column should be at regular interval of either step 1 or 2 or 3 or so on. Typically step 2 
# is expected given that the 1st column is that of SNPs and the other of DIPs.
#Author: Abhishek Narain Singh
#Date: 5th May 2019
#Execution: python3 <ProgramFilename> <InputData> <OutputFastaDipFileName> <DivergenceScoreFilename> <steps in inputFile to look for DIVs>
#Example$ python3 multiColDIPsDiv.py multiColumnSplitSample.csv multiFastaDIPs DivergenceScoreFile 2
################################################################################################################################################
import numpy as np
import pandas as pd
import sys
import subprocess

dataset = pd.read_csv(sys.argv[1], low_memory=False) #Read the multi-column splitted SNP and DIPs data file such as multiColumnSPlitSample.csv

def DIPsDiv(col): #Function to calculate the diversion of DIPs
    data = dataset.iloc[:,col]
    x=data.values
    #File output name
    outputFileName = "./readwrite/" + sys.argv[2] + str(col) #As they will be very many, it would make sense to put them in a rough folder
    #Divergence Output File Name
    divergenceFileName = "./readwrite/" + sys.argv[3] + str(col)
    counter = -1 #This keeps track of the index number
    linesAdded = 0 #This will help us decide if we need to create Output file
    for i in x: #Scanning through all the row values in column
        counter +=1
        if ((i=="0.0") or (i=="0") or (i == 0) or (i == 0.0)): #This ensures that irrespective of datatype as string or float or int, we can make comparison
            pass
        else:
            linesAdded +=1 #Keeping track of number of DIPs added to list
            if (linesAdded == 1): #Now that we have at least one line, we can open file for putting multiFasta lines and to calculate the divergent score
                fileOutput = open(outputFileName, "w")
                y=np.array(())
                index=np.array(())
            #This is always done, Since we have more lines to add we just append the lines
            y=np.append(y,i) #Appending the DIPs values
            index=np.append(index,counter)
            #Output the header
            fileOutput.write(">" + str(counter) + "\n")
            #Here goes the DIP sequence
            fileOutput.write(str(i) + "\n")
    #Check if there are lines added and so the downstream processing of multiFasta file needs to be done
    if (linesAdded > 0):
        #Closing file for multiFasta 
        fileOutput.close()
        #Open the output file to store the divergence score
        fDivergence = open(divergenceFileName,"w")
        #Process it by T_COFFEE
        subprocess.call(["t_coffee", "-other_pg", "seq_reformat", "-in", outputFileName, "-output", "sim_idscore"], stdout=fDivergence )
        fDivergence.close()
		
#Main code from here - Now we call the DIPsDiv function
#In future a parallel implementation of this can be done in case each function call is taking a lot of time (Currently that does not seem to be the case)
for i in range(1,len(dataset.columns),int(sys.argv[4])):
    DIPsDiv(i)

#DIPsDiv(3)

