
import pandas as pd  

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tag import pos_tag
from nltk.tag import StanfordNERTagger
import nltk.data
import re 

from gensim import models
from sklearn.cluster import KMeans

import os
print os.environ.get('CLASSPATH')

#For reading sentences from the training module and saving it to a list Query1
query1=[]
with open("Desktop/trainingdata.txt", "r") as myfile:
    query1.append(myfile.read())
    myfile.close()

print query1

##############################################         CLEANING           ##########################################################################
#Removing Punctuations and converting to Lower Case
def remove_punctuations(text):
    wordlist=[]    
    sentences=sent_tokenize(text)
    for sentence in sentences:
        #words=word_tokenize(sentence)
        words=[word for word in sentence.split()]               
        words=[re.sub(r'[^\w\s]','',word) for word in words]     
        wordlist.append(words)
    return wordlist


#Keeping only Nouns from the sentence
def descriptive_words(words):
    tagged_word=[]
    meaningful_words=[]    
    tags=['NN','NNS','NNP']    

    for word in words:
        tagged_word.append(pos_tag(word))
    for words in tagged_word:
        for word in words:            
            if word[1] in tags:
                        meaningful_words.append(word[0])
 
    return meaningful_words        
                

#Using NER Tagger to find Name, Organisation, Entity
def remove_names(text):
    meaningful_words=[]
    tagged_word=[]
    tags=['LOCATION', 'ORGANIZATION', 'PERSON']    
    st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
    
    for sentence in text:
        tagged_word.append(st.tag(sentence))

    for words in tagged_word:
        for word in words:
            if word[1] in tags:
                meaningful_words.append(word[0])
    return meaningful_words
    



#Cleaning of Sentences in Query1     
words_req=[]

for query in query1:
    #Removing Punctuations and changing to lower space
    letters_only = remove_punctuations(query)                       

    #5 Read only NOuns,Pronouns,interjections (descriptive words)  
    meaningful_words=descriptive_words(letters_only)


    #6 Keeping Time, Location, Organization, Person, Money, Percent, Date using NER   
    removed_words=remove_names(letters_only)    
        

    #6.Stemming using Porter Stemmer,Lemming can also be used check which is more efficient
    st=PorterStemmer()
    #stemmed_words=[st.stem(words) for words in meaningful_words+removed_words]   
    stemmed_words=[words for words in meaningful_words+removed_words] 
    words_req.append(stemmed_words)
  
print words_req     

###########################################          CLUSTERING            ###############################################################

#Changing the words into Vectors so that Clusters can be formed out of them
vec=[]
word_vec = models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print word_vec["queen"]

for sentence in words_req:
    for words in sentence:
        try:
            vec.append(word_vec[words])
        except:
            print "error",words

print "Vectors Formed"

#Creating Clusters
kmeans=KMeans(init='k-means++',n_clusters=3)
kmeans.fit_predict(vec)

print "Clustering Done"
print kmeans

#############################################          SEARCHING             ##############################################################

#Searching for only those Sentences from the training module which belongs to the vendor we are searching for DELL in this case and save it in Query3
query2=[]
with open("Desktop/trainingdata.txt", "r") as myfile:
    query2.append(myfile.read())
    myfile.close()

print query2
print "Type",type(query2)

query3=[]
for query in query2:
    letters_only = remove_punctuations(query)

    removed_words=remove_names(letters_only) 
    
    if "DELL" in removed_words:
        query3.append(query)


#############################################          PREDICTING             ##############################################################

#Cleaning of words belonging to the vendor in order to change them to vector and apply Clustering to it     
words_req=[]

for query in query3:
    #Removing Punctuations and changing to lower space
    letters_only = remove_punctuations(query)                       

    #5 Read only NOuns,Pronouns,interjections (descriptive words)  
    meaningful_words=descriptive_words(letters_only)


    #6 Keeping Time, Location, Organization, Person, Money, Percent, Date using NER   
    removed_words=remove_names(letters_only)    
        

    #6.Stemming using Porter Stemmer,Lemming can also be used check which is more efficient
    #st=PorterStemmer()
    #stemmed_words=[st.stem(words) for words in meaningful_words+removed_words]
    stemmed_words=[words for words in meaningful_words+removed_words]   
    words_req.append(stemmed_words)
  
print words_req 

vecs=[]
words=[]


#lChanging the words found into vectors
for sentence in words_req:
    for word in sentence:
        try:
            word_transformed=word_vec[word]
            vecs.append(word_transformed)
            words.append(word)
        except:
            print "error",word

print "Vector Formed"
print words

#Now Predicting to which Cluster the words describing the Vendor belongs to 
cluster_data={'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0}
cluster_words={'0':[],'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}
for pos,vec in enumerate(vecs):
    cluster_no=kmeans.predict([vec])
    print "Cluster No",cluster_no
    cluster_data[str(cluster_no[0])]+=1
    cluster_words[str(cluster_no[0])].append(words[pos])

print "Cluster matrix formed"
print cluster_data
print cluster_words

#find cluster which contains maximum number of words, print the top 7 (which occured the most in the training set for that vendor) from that cluster
max_cluster=max(cluster_data, key=cluster_data.get)
imp_words_1=set(cluster_words[max_cluster])

#Findind the 2nd max cluster,print the top 3 (which occured the most in the training set for that vendor) from that cluster
cluster_data[max_cluster]=0
max_cluster=max(cluster_data, key=cluster_data.get)
imp_words_2=set(cluster_words[max_cluster])

print imp_words_1
print imp_words_2