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

path = "./artificial/"
dataclarabrut = arff.loadarff(open(path+"xclara.arff", 'r'))
dataclara = np.array([[x[0],x[1]] for x in dataclarabrut[0]])

datachainbrut = arff.loadarff(open(path+"chainlink.arff", 'r'))
datachain = np.array([[x[0],x[1], x[2]] for x in datachainbrut[0]])


##################################################################
# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            ")
f0clara = dataclara[:,0] # tous les élements de la première colonne
f1clara = dataclara[:,1] # tous les éléments de la deuxième colonne


f0chain = datachain[:,0] # tous les élements de la première colonne
f1chain = datachain[:,1] # tous les éléments de la deuxième colonne
f2chain = datachain[:,2] # tous les éléments de la deuxième colonne
#print(f0)
#print(f1)

plt.scatter(f0clara, f1clara, s=8)
plt.title("Donnees initiales clara")

plt.show()
"""
plot = plt.axes(projection='3d')
plot.scatter3D(f0chain, f1chain, f2chain, c=f2chain, cmap='viridis')
plt.title("Donnees initiales chain")
plt.show()
"""

########################################################################
# AUTRES VISUALISATION DU JEU DE DONNEES
# (histogrammes par exemple,)
# But : essayer d'autres types de plot 
########################################################################

########################################################################
# STANDARDISER ET VISUALISER 
# But : comparer des méthodes de standardisation, ...
########################################################################
