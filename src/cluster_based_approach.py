import numpy as np
import scipy as sc
import math as mt
from scipy import cluster as clst
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from IPython.display import clear_output
from tqdm import tqdm


#  ***************************************************************************************
# * The methods in this class were originally created/coded by Sara Rajee. 
# * Repo Title:clusterbased_isotropy_enhancement
# * Repo URL :
# *   https://github.com/Sara-Rajaee/clusterbased_isotropy_enhancement
# *
# ***************************************************************************************/

class cluster_based_approach():
    
    def calculate_isotropy(representations):
        eig_values, eig_vectors = np.linalg.eig(
            np.matmul(np.transpose(representations),representations))
        
        max_f = -mt.inf
        min_f =  mt.inf

        for i in range(eig_vectors.shape[1]):
            f = np.matmul(representations, np.expand_dims(eig_vectors[:, i], 1))
            f = np.sum(np.exp(f))

            min_f = min(min_f, f)
            max_f = max(max_f, f)

        isotropy = min_f / max_f

        return isotropy
    
    
    def cluster_based(representations, n_cluster: int, n_pc: int):

        centroid, label=clst.vq.kmeans2(representations, n_cluster, minit='points',
                                  missing='warn', check_finite=True)
        cluster_mean=[]
        for i in range(max(label)+1):
            sum=np.zeros([1,768]);
            for j in np.nonzero(label == i)[0]:
                sum=np.add(sum, representations[j])
            cluster_mean.append(sum/len(label[label == i]))

        zero_mean_representation=[]
        for i in range(len(representations)):
            zero_mean_representation.append((representations[i])-cluster_mean[label[i]])

        cluster_representations={}
        for i in range(n_cluster):
            cluster_representations.update({i:{}})
        for j in range(len(representations)):
            if (label[j]==i):
                cluster_representations[i].update({j:zero_mean_representation[j]})

        cluster_representations2=[]
        for j in range(n_cluster):
            cluster_representations2.append([])
            for key, value in cluster_representations[j].items():
                cluster_representations2[j].append(value)

        cluster_representations2=np.array(cluster_representations2)

        model=PCA()
        post_rep=np.zeros((representations.shape[0],representations.shape[1]))

        for i in range(n_cluster):
            model.fit(np.array(cluster_representations2[i]).reshape((-1,768)))
            component = np.reshape(model.components_, (-1, 768))

            for index in cluster_representations[i]:
                sum_vec = np.zeros((1, 768))

                for j in range(n_pc):
                        sum_vec = sum_vec + np.dot(cluster_representations[i][index],
                                np.transpose(component)[:,j].reshape((768,1))) * component[j]
                
                post_rep[index]=cluster_representations[i][index] - sum_vec

        clear_output()

        return post_rep
    

    def mean_pooling(inp_representations, representation_dev):
    # calculating sentence representations by averaging
        sum_index = 0
        sent_representations = []
        for i in range(len(representation_dev)):
            sent_representations.append(np.mean(inp_representations[sum_index: sum_index 
                                                                    + (len(representation_dev[i]))],axis=0))
            sum_index = sum_index + len(representation_dev[i])
            
        return sent_representations
