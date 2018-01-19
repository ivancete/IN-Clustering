# -*- coding: utf-8 -*-
"""
Autor:
    Iván Rodríguez Millán.
Fecha:
    18 de Diciembre del 2017.
Contenido:
    Uso de algoritmos de Clustering con diferentes librerías.
    Inteligencia de Negocio.
    Grado en Ingeniería Informática.
    Universidad de Granada.
"""

'''
Documentación sobre clustering en Python:
    http://scikit-learn.org/stable/modules/clustering.html
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
    http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
'''

import time
import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import pylab
import numpy as np

import sklearn.cluster as cluster

#Saco el dendrograma usando scipy, que realmente vuelve a ejecutar el clustering jerárquico
from scipy.cluster import hierarchy

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn import preprocessing
from sklearn import metrics
from sklearn import preprocessing

from math import floor

##########################################################################################################
#Función para extraer las medias y desviaciones típicas de los valores de cada cluster para cada variable#
##########################################################################################################
def calcularMediaStd(cluster):
    vars = list(cluster)
    vars.remove('cluster')
    return dict(np.mean(cluster[vars],axis=0)), dict(np.std(cluster[vars],axis=0))

def DFClusterConMedias(dataFrame):

    listaClusters = list(set(dataFrame['cluster']))

    DFMedia = pd.DataFrame()
    DFStd = pd.DataFrame()

    for cluster_n in listaClusters:
        cluster_i = dataFrame[dataFrame['cluster'] == cluster_n]
        DicMedia, DicStd = calcularMediaStd(cluster=cluster_i)
        auxDFMedia = pd.DataFrame(DicMedia,index=[str(cluster_n)])
        auxDFStd = pd.DataFrame(DicStd,index=[str(cluster_n)])
        DFMedia = pd.concat([DFMedia, auxDFMedia])
        DFStd = pd.concat([DFStd, auxDFStd])

    return DFMedia, DFStd

##########################################################################################################
#Función para modificar los nombres de las columnas de los dataframes con las medias y las desviaciones#
##########################################################################################################
def DFClusterColumnasMediasNuevas(dataFrame):
    listaColumnas = list(dataFrame.columns.values)
    nuevasColumnas = []

    for nombre in listaColumnas:
        nombreNuevo = nombre.replace('TOT_','') + '_MED'
        nuevasColumnas.append(nombreNuevo)
    dataFrame.columns = nuevasColumnas


def DFClusterColumnasStdNuevas(dataFrame):
    listaColumnas = list(dataFrame.columns.values)
    nuevasColumnas = []

    for nombre in listaColumnas:
        nombreNuevo = nombre.replace('TOT_','') + '_STD'
        nuevasColumnas.append(nombreNuevo)
    dataFrame.columns = nuevasColumnas

############################################################################################################
#Función para concatenar los dataframes con las medias y las desviaciones y así tenerlo junto#
############################################################################################################
def DFClusterConcatMediasStd(dataframe1, dataframe2):
    DFNuevo = pd.concat([dataframe1, dataframe2], axis=1)
    return DFNuevo

###############################################################################################
#Función para almacenar todas las métricas producidas por los algoritmos en un único dataframe#
###############################################################################################
def DFValoresAlgoritmos(algoritmo, tiempo, nClusters, CH, SC, DFTodosDatos):
    df1 = pd.DataFrame({'Algoritmo':[algoritmo],
                    'N.Clusters':[int(nClusters)],
                    'Tiempo':[tiempo],
                    'CH':[CH],
                    'SH':[SC]})

    return df1

#######################################
#Función para pintar un scatter matrix#
#######################################
def PintarScatterMatrix(DFclusterSinOutliersAux, scatter_dir, nombreAlgoritmo, casoEstudio):
    plt.figure()

    variables = list(DFclusterSinOutliersAux)
    variables.remove('cluster')
    sns_plot = sns.pairplot(DFclusterSinOutliersAux, vars=variables, hue="cluster", palette='Paired',
                            plot_kws={"s": 25},
                            diag_kind="hist")  # en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);

    plt.savefig(scatter_dir + nombreAlgoritmo+ "Variacion-1-" + casoEstudio)
    plt.close()

################################
#Función para pintar un heatmap#
################################
def PintarHeatmap(DFMediasNormal, heatmap_dir, nombreAlgoritmo, casoEstudio, clusters_restantes):
    plt.figure()

    # Configuramos el tamaño de la gráfica para que nos quepan todas las variables del eje X.
    plt.subplots(figsize=(20, 10))

    sns.heatmap(data=DFMediasNormal, annot=True, linewidths=0.5, yticklabels=clusters_restantes, cmap='YlGn')

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.savefig(heatmap_dir + nombreAlgoritmo+ "Variacion-1-" + casoEstudio)
    plt.close()

###############################################
#Función para pintar un heatmap con dendograma#
###############################################
def PintarHeatmapConDendograma(DFMediasNormal, dendograma_dir, nombreAlgoritmo, casoEstudio, clusters_restantes):
    plt.figure()

    linkage_array = hierarchy.ward(DFMediasNormal)

    # Lo pongo en horizontal para compararlo con el generado por seaborn.
    hierarchy.dendrogram(linkage_array, orientation='left')

    sns.clustermap(DFMediasNormal, method='ward', col_cluster=False, figsize=(20, 10),
                   yticklabels=clusters_restantes, linewidths=0.5, cmap='YlGn')

    plt.savefig(dendograma_dir + nombreAlgoritmo + casoEstudio)
    plt.close()

#################################################
#Función para pintar gráfica del método de Elbow#
#################################################
def PintarGraficaElbow(numeroClusters, errorClusters, scatter_dir):

    fig, ax = plt.subplots()
    ax.plot(numeroClusters, errorClusters)

    ax.set(xlabel='Número de Clusters', ylabel='Inertia de Cluster',
           title='Comparativa K-means')
    ax.grid()

    plt.savefig(scatter_dir + "ComparativaKMeans.png")
    plt.close()

def PintarDendograma(DFMediasNormal, dendograma_dir, nombreAlgoritmo, casoEstudio):

    linkage_array = hierarchy.ward(DFMediasNormal)
    plt.figure()
    plt.clf()
    hierarchy.dendrogram(linkage_array, orientation='left')  # lo pongo en horizontal para compararlo con el generado por seaborn

    plt.savefig(dendograma_dir + nombreAlgoritmo + "DendogramaSolo" + casoEstudio)
    plt.close()

####################
#Programa principal#
####################

if __name__ == "__main__":
    
    accidentes = pd.read_csv('accidentes_2013.csv')
    
    #___________________________________________________________________________________
    
    #DATASET: Accidentes ocurridos en la comunidad autónoma de Madrid. 14.000
    # AccidentesMadrid
    #__________________________________________________________________________________
    '''
    subset = accidentes.loc[
        accidentes.COMUNIDAD_AUTONOMA.str.contains("Madrid")
        ]

    '''
    #___________________________________________________________________________________
    
    #DATASET: Accidentes ocurridos en Andalucía. 13.944
    # AccidentesAndalucia
    #__________________________________________________________________________________
    '''
    subset = accidentes.loc[
        accidentes.COMUNIDAD_AUTONOMA.str.contains("Andalucía")
        ]
    '''
    #___________________________________________________________________________________
    
    #DATASET: Accidentes con conlisión entre vehículos para el mes de Diciembre. 4.047
    # ColisionVehiculosDiciembre
    #__________________________________________________________________________________
    '''NO VOLVER HA HACER.
    subset = accidentes.loc[
        (accidentes.TIPO_ACCIDENTE.str.contains("Colisión de vehículos"))
        &(accidentes['MES'] == 12)
        ]
    '''
    #_____________________________________________________________________________________

    #DATASET: Accidentes con conlisión entre vehículos para el mes de Agosto.4.275
    # ColisionVehiculosAgosto
    #____________________________________________________________________________________
    '''NO VOLVER HA HACER.
    subset = accidentes.loc[
        (accidentes.TIPO_ACCIDENTE.str.contains("Colisión de vehículos"))
        &(accidentes['MES'] == 6)
        ]
    '''
    #_____________________________________________________________________________________

    #DATASET: Accidentes en zonas urbanas y vías urbanas. 10.000
    # ZonaUrbanaAtropellos
    #_____________________________________________________________________________
    '''
    subset = accidentes.loc[
        (accidentes.TIPO_ACCIDENTE.str.contains("Atropello"))
        &(accidentes.ZONA_AGRUPADA.str.contains("VÍAS URBANAS"))
        &(accidentes.ZONA.str.contains("ZONA URBANA"))
        ]
    '''
    #_____________________________________________________________________________

    #DATASET: Accidentes con conlisiones entre vehículos en zonas urbanas y vías urbanas. 30.705
    # ZonaUrbanaVehiculos
    #_____________________________________________________________________________
    '''
    subset = accidentes.loc[
        (accidentes.TIPO_ACCIDENTE.str.contains("Colisión de vehículos"))
        &(accidentes.ZONA_AGRUPADA.str.contains("VÍAS URBANAS"))
        &(accidentes.ZONA.str.contains("ZONA URBANA"))
        ]
    '''
    #_____________________________________________________________________________


    #DATASET: Accidentes con conlisiones entre vehículos en un trazado con curva suave. 2.542
    # TrazadoCurvaSuave.png
    
    #_____________________________________________________________________________
    '''
    subset = accidentes.loc[
        (accidentes.TIPO_ACCIDENTE.str.contains("Colisión de vehículos"))
        &(accidentes.TRAZADO_NO_INTERSEC.str.contains("CURVA SUAVE"))
        ]
    '''
    #_____________________________________________________________________________


    #DATASET: Accidentes con conlisiones entre vehículos en un trazado con curva fuerte. 1.143
    # TrazadoCurvaFuerte.png
    
    #_______________________________________________________________________________

    subset = accidentes.loc[
        (accidentes.TIPO_ACCIDENTE.str.contains("Colisión de vehículos"))
        &(accidentes.TRAZADO_NO_INTERSEC.str.contains("CURVA FUERTE"))
        ]

    #_____________________________________________________________________________

    #DATASET: Accidentes con conlisiones entre vehículos en un trazado con curva fuerte. 20.000
    # TrazadoRecta.png
    
    #_______________________________________________________________________________
    '''
    subset = accidentes.loc[
        (accidentes.TIPO_ACCIDENTE.str.contains("Colisión de vehículos"))
        &(accidentes.TRAZADO_NO_INTERSEC.str.contains("RECTA"))
        ]
    '''
    #_____________________________________________________________________________

    
    #seleccionar variables de interés para clustering
    usadas = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
    usadasOtra = ['TOT_HERIDOS_GRAVES','TOT_HERIDOS_LEVES', 'TOT_MUERTOS','TOT_VEHICULOS_IMPLICADOS', 'TOT_VICTIMAS']

    X = subset[usadas]

    print("Tamaño del conjunto de datos extraído: ",len(X), end='\n')

    tamanio = len(X)

    #Indicamos la semilla al conjunto actual de dataset.
    X = X.sample(tamanio, random_state=123456789)
    
    #Inicialización de los distintos algoritmos de la librería SKLEARN.
    X_normal = preprocessing.normalize(X, norm='l2')
    
    k_means = cluster.KMeans(init='k-means++', n_clusters=4, n_init=5)

    k_means2 = cluster.KMeans(init='k-means++', n_clusters=5, n_init=5)

    k_means3 = cluster.KMeans(init='k-means++', n_clusters=6, n_init=5)

    k_means4 = cluster.KMeans(init='k-means++', n_clusters=7, n_init=5)

    k_means5 = cluster.KMeans(init='k-means++', n_clusters=8, n_init=5)

    k_means6 = cluster.KMeans(init='k-means++', n_clusters=9, n_init=5)
    #_____________________________________________________________________________
    
    mbkm = cluster.MiniBatchKMeans(init='k-means++',n_clusters=4, n_init=5, 
                                  max_no_improvement=10, verbose=0)

    #_____________________________________________________________________________

    ward = cluster.AgglomerativeClustering(n_clusters=20,linkage='ward')

    #_____________________________________________________________________________

    #dbscan = cluster.DBSCAN(eps=0.1,min_samples=10)
    dbscan = cluster.DBSCAN(eps=0.5, min_samples=1000)

    #_____________________________________________________________________________

    #birch = cluster.Birch(threshold=0.1, n_clusters=4)
    birch = cluster.Birch(threshold=0.1, n_clusters=3)

    #_____________________________________________________________________________
    
    #spectral = cluster.SpectralClustering(n_clusters=4)
    spectral = cluster.SpectralClustering(n_clusters=9)

    #_____________________________________________________________________________

    #bandwidth = cluster.estimate_bandwidth(X_normal, quantile=0.2, n_samples=tamanio)
    meanshift = cluster.MeanShift( bin_seeding=True)

    #_____________________________________________________________________________

    comparativaKMeans = False
    '''
    clustering_algorithms = (("K-means", k_means),
                            ("K-means", k_means2),
                            ("K-means", k_means3),
                            ("K-means", k_means4),
                            ("K-means", k_means5),
                            ("K-means", k_means6)                      
    )
    '''

    #_____________________________________________________________________________

    clustering_algorithms = (("K-means", k_means6),
                             #("MiniBatchKMeans", mbkm),
                             #("Birch", birch),
                             #("Ward", ward),
                             ("DBSCAN", dbscan),
                             #("MeanShift",meanshift),
                             ("Spectral", spectral)
    )
    #_____________________________________________________________________________

    casoEstudio = "TrazadoCurvaFuerte.png"
    script_dir = os.getcwd()
    heatmap_dir = os.path.join(script_dir, 'heatmap/')
    dendograma_dir = os.path.join(script_dir, 'dendograma/')
    scatter_dir = os.path.join(script_dir, 'scattermatrix/')

    #Variables para almacenar datos de interés de los algoritmos para mostrarlos
    #posteriormente.

    errorClustersKmean = []
    clusterKmean =[]
    tiempoPorAlgoritmo = {}

    DFTodosDatos = pd.DataFrame(columns=['Algoritmo', 'N.Clusters','Tiempo', 'CH', 'SH'])
    
    #_____________________________________________________________________________

    #Ejecución de los distintos algoritmos previamente inicializados.
    print('_______________________________________________________')

    for name, algorithm in clustering_algorithms:
        #print('{:19s}'.format(name), end='')
        t = time.time()
        clusterPredict = algorithm.fit_predict(X_normal)
        tiempo = time.time() - t
        numeroClusterInicial = len(set(clusterPredict))

        #Esto nos sirve para el método de Elbow
        if (name is 'K-means') and (comparativaKMeans):
            #print("Inertia: {:.5f}".format(algorithm.inertia_))
            clusterKmean.append(numeroClusterInicial)
            errorClustersKmean.append(algorithm.inertia_)

        #Sirve para añadir una nueva columna llamada cluster que contiene los 
        # datos del clusterPrecit se convierte la asignación de clusters a DataFrame
        columnaClusters = pd.DataFrame(clusterPredict,index=X.index,columns=['cluster'])
            
        #se añade como columna a X
        datasetConCluster = pd.concat([X, columnaClusters], axis=1)

        #Filtro quitando los elementos (outliers) que caen en clusters muy pequeños en el jerárquico
        minimoTama = 3
        DFclusterSinOutliers = datasetConCluster[datasetConCluster.groupby('cluster').cluster.transform(len) > minimoTama]
        numeroClusterPostFiltrado = len(set(DFclusterSinOutliers['cluster']))
        clusters_restantes = list(set(DFclusterSinOutliers['cluster']))


        #Valor perteneciente al número de cluster pre filtrado.
        #print("& {:3.0f} ".format(numeroClusterInicial), end='')
        #Valor perteneciente al tiempo en segundos que tarda el algoritmo.
        #print("& {:6.2f} & ".format(tiempo), end='')
        if (numeroClusterPostFiltrado>1) and (name is not 'Ward'):
            metric_CH = metrics.calinski_harabaz_score(X_normal, clusterPredict)
            metric_SC = metrics.silhouette_score(X_normal, clusterPredict, metric='euclidean', 
            sample_size=floor(0.1*len(X)), random_state=123456)
        else:
            metric_CH = 0
            metric_SC = 0
        
        #Valor perteneciente al CH.
        #print("{:8.3f} & ".format(metric_CH), end='')
        #Valor perteneciente al SH.
        #print("{:.5f}".format(metric_SC), end='\n')
        print("De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(numeroClusterInicial,numeroClusterPostFiltrado,minimoTama,len(X),len(DFclusterSinOutliers)))

        DFclusterSinOutliersAux = DFclusterSinOutliers

        DFMedias, DFStd = DFClusterConMedias(DFclusterSinOutliersAux)

        DFMediasAux = DFMedias

        df = DFValoresAlgoritmos(name, tiempo, numeroClusterInicial, metric_CH, metric_SC, DFTodosDatos)

        DFTodosDatos = pd.concat([DFTodosDatos,df])

        DFMediasNormal = preprocessing.normalize(DFMedias, norm='l2')
        DFMediasNormal = pd.DataFrame(DFMediasNormal, columns=usadasOtra)


        '''
        if (name != 'Ward'):

            PintarScatterMatrix(DFclusterSinOutliersAux, scatter_dir, name, casoEstudio)
        '''

        '''
        if ((casoEstudio == 'TrazadoCurvaSuave.png' or casoEstudio == 'TrazadoCurvaFuerte.png' or casoEstudio == 'TrazadoRecta.png'
            or casoEstudio == 'ZonaUrbanaVehiculos.png' or casoEstudio == 'AccidentesMadrid.png' or casoEstudio == 'AccidentesAndalucia.png'
            or casoEstudio == 'ZonaUrbanaVehiculos.png' or casoEstudio == 'ZonaUrbanaAtropellos.png')
            and name != 'Ward'):

            PintarHeatmap(DFMediasNormal, heatmap_dir, name, casoEstudio, clusters_restantes)
        '''
        if (((casoEstudio == 'ZonaUrbanaAtropellos.png') or (casoEstudio == 'ZonaUrbanaVehiculos.png')) and (name == 'Ward')):

            PintarHeatmapConDendograma(DFMediasNormal, dendograma_dir, name, casoEstudio, clusters_restantes)
            PintarDendograma(DFMediasNormal, dendograma_dir, name, casoEstudio)

        print(DFMedias.to_latex())

    print(DFTodosDatos.to_latex(index=False))


    if (comparativaKMeans):
        PintarGraficaElbow(clusterKmean, errorClustersKmean, scatter_dir)

