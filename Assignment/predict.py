from gensim import models
from sklearn.cluster import KMeans


wordvec = models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

vecs=[]
words=[]


#list of word tokenized sentences
for sentence in words_req:
	for word in sentence:
		word_transformed=wordvec[word]
		vecs.append(word_transformed)
		words.append(word)


#returns a list of vectors for the particular item

cluster_data={'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0}
cluster_words={'0':[],'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}
for pos,vec in iter(vecs):
	cluster_no=kmeans.predict(vec)
	cluster_data[cluster_no]+=1
	cluster_words[cluster_no].append(words[pos])


#find max cluster do set on that cluster_words and print the top 10 from that cluster
max_cluster=max(cluster_data, key=cluster_data.get)
imp_words_1=set(cluster_words[max_cluster])

#Findind the 2nd max cluster
cluster_data[max_cluster]=0
max_cluster=max(cluster_data, key=cluster_data.get)
imp_words_2=set(cluster_words[max_cluster])

