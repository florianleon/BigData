# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 23:08:23 2021

@author: huguet
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn import preprocessing

##################################################################
# READ a data set (arff format)

# Parser un fichier de données au format arff
# datanp est un tableau (numpy) d'exemples avec pour chacun la liste 
# des valeurs des features

# Note 1 : 
# dans les jeux de données considérés : 2 features (dimension 2 seulement)
# t =np.array([[1,2], [3,4], [5,6], [7,8]]) 
#
# Note 2 : 
# le jeu de données contient aussi un numéro de cluster pour chaque point
# --> IGNORER CETTE INFORMATION ....
#    2d-4c-no9.arff

path = './artificial/'
databrut = arff.loadarff(open(path+"xclara.arff", 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])
#print(databrut)
#print(datanp)

##################################################################
# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            ")
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne
ground_truth = np.array([[x[0],x[1],x[2]] for x in databrut[0]])[:,2]

plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()

##################################################################
# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée (données init)")

def find_max(liste):
    max_index, max_value = max(enumerate(liste), key=lambda x: x[1])
    return max_index

def find_min(liste):
    min_index, min_value = min(enumerate(liste), key=lambda x: x[1])
    return min_index

def plot_metrics(Y, title):
    plt.plot(range(2,10), Y)
    plt.title(title)
    plt.show()

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

def get_best_k():
    inerts = []
    silhs = []
    homs = []
    v_meas = []
    model_kms = []
    for k in range(2,10):
        model_km = cluster.KMeans(n_clusters=k, init='k-means++')
        model_km.fit(datanp)
        labels_km = model_km.labels_
        inerts.append(model_km.inertia_)
        silhs.append(metrics.silhouette_score(datanp, model_km.labels_, metric='euclidean'))
        homs.append(metrics.homogeneity_score(ground_truth, labels_km))
        v_meas.append(metrics.v_measure_score(ground_truth, labels_km))
        model_kms.append(model_km)

    best_silh = find_max(silhs)
    best_v_meas = find_max(v_meas)
    inerts = preprocessing.scale(inerts)
    best_inert = find_best_index(inerts, 0.0)
    best_homs = find_best_index(homs, 1.0)
    # Calcul de la moyenne des metrics
    mean_k = (best_inert + best_silh + best_v_meas + best_homs) / 4
    # On rattrape le décalage d'index 
    best_k = round(mean_k) + 2
    plot_metrics(inerts, "Inertie")
    """
    plot_metrics(inerts, "Inertie")
    plot_metrics(silhs, "Silhouette")
    plot_metrics(homs, "Homogénéité")
    plot_metrics(v_meas, "V-measure") 
    """  

    print("best k is = ", best_k)
    print("Best inert", best_inert + 2)
    print("Best silh", best_silh + 2)
    print("Best v_meas", best_v_meas + 2)
    print("Best homs", best_homs + 2)
    print("------------------------------------------------------")

    return best_k, model_kms[best_k-2]

def run_k_means():
    tps1 = time.time()
    k, model_km = get_best_k()
    tps2 = time.time()
    # Nb iteration of this method
    iteration = model_km.n_iter_
    labels_km = model_km.labels_
    plt.scatter(f0, f1, c=labels_km, s=8)
    plt.title("Données (init) après clustering")
    plt.show()
    print("nb clusters =",k,", nb iter =",iteration, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")

run_k_means()
########################################################################
# TESTER PARAMETRES METHODE ET RECUPERER autres métriques
########################################################################