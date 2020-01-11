# MDLWAS
Machine &amp; Deep Learning Wide Association Study

Univariate methods for association of the genomic variations with the end-or-endo-phenotype has been widely used by means of software tools such as snptest and p-link. In addition to encoding the SNPs, we introduce a novel method to encode the structural variations in genome such as the deletions and insertions polymorphism (DIPs) that can then be used downstream by artificial intelligence methods as an independent feature variable value to predict the endo-or-end phenotype. We conducted a complete all genomic variants association with the phenotype using deep learning and other machine learning techniques such as logistic regression, naïve bayes, gradient boosting, bagging, and adaboost. As a combination of the encoding scheme on a simulated DNA variation dataset, we were able to get near 100% accuracy using deep learning, while up to 82% accuracy in prediction was achieved using other machine learning algorithms. Deep learning script, however, took much more time and computational resources to determine the optimum parameters. We also make the source-codes available as a suit of software under the name MDLWAS.  Given that simulated data are not real cases as they have random component, MDLWAS is positioned to perform better even for algorithms other than deep learning for real case genomic data. 

###################################Generate Sample Genotype with DIPs####################################################################\
$>python3 genSampleData.py <no. of genotypes columns> <no. of individuals or rows> <frequency of one of how many> <max limit size of insertions> <outputFileName>\
Example:\
-bash-4.2$ python3 genSampleData.py 40 50 28 11 simulatedSNPLowDIPs.txt\
#########################################################################################################################################\

