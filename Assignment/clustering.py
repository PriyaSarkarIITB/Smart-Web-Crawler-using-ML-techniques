from gensim import models
from sklearn.cluster import KMeans
import cleaning.py

wordvec = models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
vecs=[]

#list of word tokenized sentences
for sentence in words_req:
	for words in sentence:
		word=wordvec[words]
		vecs.append(word)
#returns a list of vectors for the particular item

kmeans = KMeans(init='k-means++', n_clusters=3, n_iter=20)
kmeans.fit(vecs)

