# Smart-Web-Crawler-using-ML-techniques

OverAll Procedure-
Step 1: Crawling over the Internet, and saving information about retail companies into training.txt
file.
Step 2: Cleaning the data-The data is in form of paragraphs and sentences. It is tokenized and
from that only those tokens are chosen which potentially contains information about the
vendor. For example, only the Nouns are kept and words which are tagged under Name,
Organization, Location or Entity are kept.
Step 3: At this step, the cleaned words are changed into vectors via Word2Vec. Word2Vec
changes each word into a vector containing an array of numbers such that mathematical
computations like clustering can be done on textual data. Now these words are used for
forming Clusters using K Means Clustering.
Step 4: Now we input a vendor (whose details we need to find out), We crawl over the training.txt
file and extract only those paragraphs which are pertaining to the input vendor. We do it via
NER Tagger.We check the Organization for each paragraph in our training set and in which
paragraphs the (Organization==Input Vendor Name) and if this has happened more than
thrice it is considered as a suitable paragraph for the Vendor.This is our Test Data
Step 5: Now we use our Kmeans model created in Step 3 and determine to which cluster the
words in our Test Data belongs to. Thus we create a Cluster frequency matrix like Cluster
1:5 words, Cluster 2:1 words,Cluster 3:0 words. Now we select the top 7 words (which has
occurred maximum number of times in the Test Data) from the cluster containing maximum
number of words and top 3 words from the 2nd max cluster.
