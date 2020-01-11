# MDLWAS
Machine &amp; Deep Learning Wide Association Study

Univariate methods for association of the genomic variations with the end-or-endo-phenotype has been widely used by means of software tools such as snptest and p-link. In addition to encoding the SNPs, we introduce a novel method to encode the structural variations in genome such as the deletions and insertions polymorphism (DIPs) that can then be used downstream by artificial intelligence methods as an independent feature variable value to predict the endo-or-end phenotype. We conducted a complete all genomic variants association with the phenotype using deep learning and other machine learning techniques such as logistic regression, naïve bayes, gradient boosting, bagging, and adaboost. As a combination of the encoding scheme on a simulated DNA variation dataset, we were able to get near 100% accuracy using deep learning, while up to 82% accuracy in prediction was achieved using other machine learning algorithms. Deep learning script, however, took much more time and computational resources to determine the optimum parameters. We also make the source-codes available as a suit of software under the name MDLWAS.  Given that simulated data are not real cases as they have random component, MDLWAS is positioned to perform better even for algorithms other than deep learning for real case genomic data. 

###################Generate Sample Genotype with DIPs#########################################\
$>python3 genSampleData.py <no. of genotypes columns> <no. of individuals or rows> <frequency of one of how many> <max limit size of insertions> <outputFileName>\
Example:\
-bash-4.2$ python3 genSampleData.py 40 50 28 11 simulatedSNPLowDIPs.txt\
#######################################################################################\
Then you would split the genotype information into multiple columns of SNPs and DIPs by command below:\

For small size of the file use the serial version like below:\
#Usage$ python3 splitMultiColDIPs.py <inputFileName> <outputFileNameDesired>\
#Example$ python3 splitMultiColDIPs.py multiColumnSample.csv multiColumnSplitSample.csv\
  
Do not attempt to use this script for large files, as the process can get hung for several weeks or months without producing results.\
########################################################################################\
Then for splitting multiple columns DIPs Divergence score use multiColDIPsDiv.py\
This file extracts the DIPs from each column and puts them as a multi-fasta file. Then it calls T_coffee internally and outputs the divergence score to a divergence score file.\
Dependencies: T-Coffee, External Data for DIPS such as that below:\
0 T 0.0\
1 I AATTGGC\
Note: The DIPs column should be at regular interval of either step 1 or 2 or 3 or so on. Typically step 2 
 is expected given that the 1st column is that of SNPs and the other of DIPs.\
Execution: python3 +ProgramFilename+ +InputData+ +OutputFastaDipFileName+ +DivergenceScoreFilename+ +steps in inputFile to look for DIVs+ \
Example$ python3 multiColDIPsDiv.py multiColumnSplitSample.csv DIPsCol divergence 2\ 
##########################################################################################\
Then the scores are taken from the reverse or bottom section of the divergence files created.\
#USAGE $ python3 programname fileWithDivergenceScore fileNametoStoreSCoreAverageScore fileToRetrieveColumnNumber jumpingIndexCountSteps\
#Example: $ python3 reverseReadMulti.py divergence reverse multiColumnSplitSample.csv 2\
**Note : Only files with no blank content in terms of corresponding DIPs value for the column will be generated with their scores.\
###########################################################################################\

