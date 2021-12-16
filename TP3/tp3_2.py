import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn import metrics
import time
from sklearn.neighbors import NearestNeighbors

##################################################################
# Récupération des données

fp = "./new-data/pluie.csv"
datanp = pd.read_csv(fp, encoding="latin-1")
print(datanp.head())
#print(datanp.columns)
data1 = datanp[["JANVIERp", "FEVRIERp", "MARSp", "AVRILp", "MAIp", "JUINp", "JUILLETp", "AOUTp", "SEPTEMBREp", "OCTOBREp", "NOVEMBREp", "DECEMBREp"]]
data2 = datanp[["JANVIERnb.j.pl", "FEVRIERnb.j.pl", "MARSnb.j.pl", "AVRILnb.j.pl", "MAInb.j.pl", "JUINnb.j.pl", "JUILLETnb.j.pl", "AOUTnb.j.pl", "SEPTEMBREnb.j.pl", "OCTOBREnb.j.pl", "NOVEMBREnb.j.pl", "DECEMBREnb.j.pl"]]
data3 = datanp[['Température moyenne annuelle', 'Amplitude annuelle des températures','Insolation annuelle','Latitude', 'Longitude','Précipitations de mai à aout', 'Précipitations sept-oct']]
data = pd.concat([data1, data2, data3], axis = 1)

##################################################################
# On applique un preprocessing sur les données avant d'appliquer le PCA

# Separating out the features
x = data.values
# Standardizing the features
#x = preprocessing.StandardScaler().fit_transform(x)
x = preprocessing.MinMaxScaler().fit_transform(x)
#x = preprocessing.minmax_scale(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['Comp1', 'Comp2'])
#plt.scatter(principalDf.Comp1, principalDf.Comp2)
#plt.show()
mainDf = pd.concat([principalDf, datanp["Géographie"]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA with MinMaxScaler', fontsize = 20)
targets = ['Nord', 'Est', 'Sud', 'Ouest']
colors = ['r', 'g', 'b', 'y']
for target, color in zip(targets,colors):
    indicesToKeep = mainDf['Géographie'] == target
    ax.scatter(mainDf.loc[indicesToKeep, 'Comp1']
               , mainDf.loc[indicesToKeep, 'Comp2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

##################################################################
# Utilisation de la méthode k-means

# Données standardisées
f0 = principalDf.Comp1
f1 = principalDf.Comp2

print("------------------------------------------------------")
print("Utilisation de k-means sur les données scaled")

def find_max(liste):
    max_index, max_value = max(enumerate(liste), key=lambda x: x[1])
    return max_index

def find_min(liste):
    min_index, min_value = min(enumerate(liste), key=lambda x: x[1])
    return min_index

def compute_distance(X, Y):
    return np.sqrt(np.sum(np.square(X[0]-X[1]) + np.square(Y[0]-Y[1])))

def find_best_index(liste, cmp):
    distances = []
    best_index = 0
    for i in range(len(liste)):
        distance = compute_distance([0.0, cmp], [i, liste[i]])
        distances.append(distance)

    best_index = find_min(distances)
    return best_index

def get_best_k(datanp):
    inerts = []
    silhs = []
    model_kms = []
    for k in range(2,10):
        model_km = cluster.KMeans(n_clusters=k, init='k-means++')
        model_km.fit(datanp)
        labels_km = model_km.labels_
        inerts.append(model_km.inertia_)
        silhs.append(metrics.silhouette_score(datanp, model_km.labels_, metric='euclidean'))
        model_kms.append(model_km)

    best_silh = find_max(silhs)
    inerts = preprocessing.scale(inerts)
    best_inert = find_best_index(inerts, 0.0)
    # Calcul de la moyenne des metrics
    mean_k = (best_inert + best_silh) / 2
    # On rattrape le décalage d'index 
    best_k = round(mean_k) + 2
    print("best k is = ", best_k)
    print("Best inert", best_inert + 2)
    print("Best silh", best_silh + 2)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    return best_k, model_kms[best_k-2]

def run_k_means():
    tps1 = time.time()
    k, model_km = get_best_k(principalDf)
    tps2 = time.time()
    # Nb iteration of this method
    iteration = model_km.n_iter_
    labels = model_km.labels_
    #k=4
    print("nb clusters =",k,", nb iter =",iteration, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title(f'K-means with k = {k}', fontsize = 20)
    ax.scatter(f0, f1, c = labels, s = 50)
    ax.grid()
    plt.show()

run_k_means()

##################################################################
# Utilisation de clusters hierarchiques sur les données scaled

def get_best_agg(datanp, linkage):
    silhs = []
    dbss = []
    models = []
    
    for k in range(2,10):
        model_scaled = cluster.AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage=linkage)
        model_scaled.fit(datanp)     
        labels_scaled = model_scaled.labels_
        silhs.append(metrics.silhouette_score(datanp, labels_scaled, metric='euclidean'))  
        dbss.append(metrics.davies_bouldin_score(datanp, labels_scaled))
        models.append(model_scaled)

    best_silh = find_max(silhs)
    best_dbss = find_min(dbss)

    mean_k = (best_silh + best_dbss)/2
    best_k = round(mean_k) + 2
    print("Best silhouette : ", best_silh + 2)
    print("Best Davies-Bouldin : ", best_dbss + 2)
    print("\nbest k is = ", best_k)
    print("------------------------------------------------------")

    return best_k, models[best_k-2]

def run_with_best_k():
    aggs = ['single', 'average', 'complete', 'ward']
    for agg in aggs:
        k, model_scaled = get_best_agg(principalDf, agg)
        labels = model_scaled.labels_
        print(f"Appel Aglo Clustering '{agg}' pour une valeur de {k} determinée automatiquement")
        #With scaled data
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title(f"Aglo Clustering '{agg}' pour k = {k}", fontsize = 20)
        ax.scatter(f0, f1, c = labels, s = 50)
        ax.grid()
        plt.show()

run_with_best_k()

print("------------------------------------------------------")

def Get_distanceMean(points,minPts,previous_distanceMean):

    if (minPts < len(points)):
        nbrs = NearestNeighbors(n_neighbors=minPts).fit(points)
        distances, indices = nbrs.kneighbors(points)
        d_mean = distances.mean()
        return d_mean
    else:
        return previous_distanceMean

def KNNdist_plot(points,minPts):
    epsPlot = []
    current_distanceMean = previous_distanceMean = 0
    knee_value = knee_found = 0
    n_trainingData = 0
    for i in range (0,len(points),5):
        current_distanceMean = Get_distanceMean(points[i:],minPts,previous_distanceMean)
        df = current_distanceMean - previous_distanceMean
        
        #print("x=" + str(i) + " , df=" + str(df))
        if (df > 0.01 and i > 1 and knee_found == 0):
            knee_value = current_distanceMean
            knee_found = 1
            n_trainingData = i
            
        epsPlot.append( [i,current_distanceMean] )
        previous_distanceMean = current_distanceMean
    
    
    #Plot the kNNdistPlot
    for i in range(0, len(epsPlot)):
                plt.scatter(epsPlot[i][0],epsPlot[i][1],c='r',s=3,marker='o')
    plt.axhline(y=knee_value, color='g', linestyle='-')
    plt.axvline(x=n_trainingData , color='g', linestyle='-')
    plt.title("kNNdistPlot")
    plt.show()
    print("Knee value: x=" + str(n_trainingData) + " , y=" + str(knee_value))
    
    return knee_value


#On utilise la fonction uniquement pour determiner la distance
kv = KNNdist_plot(principalDf,3)

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

def find_best_min_samples(data, distance):
    for k in range(2,8):
        min_pts = k
        cl_pred = cluster.DBSCAN(eps=distance, min_samples=min_pts).fit_predict(data)
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
        n_noise_ = list(cl_pred).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        plt.scatter(f0, f1, c=cl_pred, s=50)
        plt.title(f"Clustering DBSCAN - Epilson={distance} - Minpt={min_pts}")
        plt.show()
    return n_clusters_

find_best_min_samples(principalDf, kv)

print("------------------------------------------------------")