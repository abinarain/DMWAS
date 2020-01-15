# DMWAS
Deep &amp; Machine learning Wide Association Study

Univariate methods for association of the genomic variations with the end-or-endo-phenotype has been widely used by means of software tools such as snptest and p-link. In addition to encoding the SNPs, we introduce a novel method to encode the structural variations in genome such as the deletions and insertions polymorphism (DIPs) that can then be used downstream by artificial intelligence methods as an independent feature variable value to predict the endo-or-end phenotype. We conducted a complete all genomic variants association with the phenotype using deep learning and other machine learning techniques such as logistic regression, naïve bayes, gradient boosting, bagging, and adaboost. As a combination of the encoding scheme on a simulated DNA variation dataset, we were able to get near 100% accuracy using deep learning, while up to 82% accuracy in prediction was achieved using other machine learning algorithms. Deep learning script, however, took much more time and computational resources to determine the optimum parameters. We also make the source-codes available as a suit of software under the name MDLWAS.  Given that simulated data are not real cases as they have random component, MDLWAS is positioned to perform better even for algorithms other than deep learning for real case genomic data. 

IMPORTANT: Please create directories by name 'readwrite', 'readwrite2' without the quotes. This is where the intermediate files will be stored. 
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
IMPORTANT: Please create directories by name 'readwrite','readwrite2' without the quotes. This is where the intermediate files will be stored. 
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
Note : Only files with no blank content in terms of corresponding DIPs value for the column will be generated with their scores.\
###########################################################################################\
The multiple files which comprises of the scores are then used to replace the DIPs in MultiColFile resulting in MultiColDIPsScore file
#Usage$ python3 ReplaceMultiColDIPs.py MultiColFile StepsWhereMultiColsAre ReverseScoreFileName \
#Example: python3 ReplaceMultiColDIPs.py multiColumnSplitSample.csv 2 reverse \

This creates file with DIPs replaced with Divergence score with the name MultiColDIPsScored.txt\
###########################################################################################\
After encoding the DIPS, or even before encoding the DIPs, once we have the SNPs and DIPs columns splitted, we can encode the SNPs. It will be best encode for the SNPs columns after the DIPs columns are encoded, using the scripts and flow above.\

Description: This script takes a Tab separated file where the 1st column is a SNP column value and iterates in gaps of 2, the file is then encoded  for each of the SNP values to get encoded. The column variable name label is added to the top appropriately by the column number underscore and the  SNP value or the value that there exists in the SNP row. Note that the separator can be changed to comma by just editing sep value.\
 Usage: python3 programName inputFileName outputFileName\
 Example: python3 encodeSNPs.py MultiColDIPsScored.txt MultiColDIPsScoredEncoded.txt\
##############################################################################################\
ExhaustiveDNN\
 
This script runs on anaconda jupyter, however, user may well extract the code and run it on shell prompt. Jupyter notebook 5.7.8 was used to test the script. This script would need you to create a directory by name model where the best score and the model corresponding to the best score will be saved. The input file for this script is MultiColDIPsScoredEncoded.txt which is the encoded file for genotype, and the Phenotype.txt file which contains the dependent y-values.\
#################################################################################################\
Machine Learning Scripts createLogitReg.py, createAdaBoost.py, createBagging.py, createGradientBoosting.py, createNaiveBayes.py\

Description: These script creates machine learning models of encoded Variant file having SNPs and/or DIPs\
#It also prints the AUC curve and CrossTab values to check for precision, false positive, true negative, etc.\ 
#It also generates a file PDValues.csv that has partial dependence of each column sorted in ascending order.\
#Usage: python3 programName EncodedFileName.txt PhenotypeFile ColumnNumberwithYvals-1 #Example 5 for 6th column\
#Example: python3 createLogitReg.py NullMafMultiColDIPsScoredEncoded.txt PhenotypeFam.txt 5 \
Here the Phenotype.txt file itself is represented in FAM format such as in PLINK tool with 1 unaffected and 2 affected, and thus named it as PhenotypeFam.txt . 
#Note: It would be useful to take the input encoded file which has been already filtered with MAF and Null values and any other discrepancy if any
#Note: the Phenotype file is space delimited and does not have header file name, The EncodedFileName is comma delimited and the header names are the 
#Note: The phenotype column variable should be of 2 values in nature, such as 1 unaffected and 2 affected, 
#and any missing phenotype should be put -9. This is as per PLINK .FAM file input format for phenotype
#column number_ character
Dependencies: python3.7.3, itertools, matplotlib needs to be installed. 
#####################################################################################################\




