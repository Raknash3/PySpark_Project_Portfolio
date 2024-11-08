'''
The pupose of this project is to calculate the TF-IDF of a word in a document. 
And then find the document that is highly relevant to a target word using the TF-IDF value.
The document with the highest TF-IDF value is considered the most relevant to the target word.
The dataset used is a subset of the Gutenberg Project dataset.
'''

#Import libraries
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

# Initialize Spark
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf = conf)

# Load documents.
rawData = sc.textFile("subset-small.tsv")
fields = rawData.map(lambda x: x.split("\t"))
documents = fields.map(lambda x: x[3].split(" "))

# Store the document names for later:
documentNames = fields.map(lambda x: x[1])

# Calculate the TF of each term in each document.
hashingTF = HashingTF(100000)  
tf = hashingTF.transform(documents)

# Calculate the IDF of each term in each document.
tf.cache()
idf = IDF(minDocFreq=2).fit(tf)

# Calculate the TF-IDF of each term in each document.
tfidf = idf.transform(tf)

# Gettysburg is that target word. Get its hash value.
gettysburgTF = hashingTF.transform(["Gettysburg"])
gettysburgHashValue = int(gettysburgTF.indices[0])

# Extract the TF*IDF score for Gettsyburg's hash value.
gettysburgRelevance = tfidf.map(lambda x: x[gettysburgHashValue])

# Zip in the document names to see which is the most relevant.
zippedResults = gettysburgRelevance.zip(documentNames)

# Print the document with the maximum TF*IDF value:
print("Best document for Gettysburg is:")
print(zippedResults.max())
