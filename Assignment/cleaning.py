
import pandas as pd  

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tag import pos_tag
from nltk.tag import StanfordNERTagger
import nltk.data
import re 

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