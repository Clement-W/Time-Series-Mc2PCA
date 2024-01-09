"""
This file contains the functions to evaluate the results of the clustering.
"""

from sklearn.metrics import confusion_matrix, adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from Mc2PCA_class import Mc2PCA

def compute_precision(C, G):
    """
    Compute the precision of clustering as described in the article.
    
    C is a list of numpy arrays, where each array represents the indices of MTS objects in the jth predicted cluster.
    G is a list of sets, where each set contains the indices of MTS objects in the ith true group.
    
    Parameters:
    - C: A list of numpy arrays representing the predicted clusters.
    - G: A list of sets representing the true groups.
    
    Returns:
    - The precision of the clustering.
    """
    N = sum(len(c) for c in C)  # Total number of MTS objects
    precision = 0.0
    
    for cj in C:
        cj_set = set(cj)
        max_intersection = max(len(cj_set.intersection(gi)) for gi in G)
        if len(cj) == 0:
            continue  # To avoid division by zero
        precision += (len(cj) / N) * (max_intersection / len(cj))
        
    return precision

def compute_recall(C, G):
    """
    Compute the recall of clustering.
    
    C is a list of numpy arrays, where each array represents the indices of MTS objects in the jth predicted cluster.
    G is a list of sets, where each set contains the indices of MTS objects in the ith true group.
    
    Parameters:
    - C: A list of numpy arrays representing the predicted clusters.
    - G: A list of sets representing the true groups.
    
    Returns:
    - The recall of the clustering.
    """
    N = sum(len(g) for g in G)  # Total number of MTS objects
    recall = 0.0
    
    for gi in G:
        gi_set = set(gi)
        max_intersection = max(len(gi_set.intersection(cj)) for cj in C)
        recall += (len(gi) / N) * (max_intersection / len(gi))
        
    return recall

def get_p_max(X, K, y_test, metric = "precision", seuil = 1e-7, max_iter = 50):
    """ 
    Compute the optimal p for Mc2PCA according to a given metric.
    """
    nbre_features = X.shape[1] # Works for DataFrame and numpy array

    idx_test = [np.where(y_test == str(i))[0] for i in range(1,np.max(y_test.astype(int))+1)]
    y_estimate = None
    values = []
    for p in range(1, nbre_features):
        model = Mc2PCA(K, p, seuil, max_iter=max_iter)
        idx_estimate, _, _ = model.fit(X)
        if metric=="precision":
            precision = compute_precision(idx_estimate, idx_test)
            values.append(precision)
        if metric=="recall":
            recall = compute_recall(idx_estimate, idx_test)
            values.append(recall)
        if metric=="ARI":
            y_estimate = np.arange(len(y_test))
            for i in range(len(idx_estimate)):
                y_estimate[idx_estimate[i]] = i + 1 
            ari_score = adjusted_rand_score(y_test.astype(int), y_estimate)
            values.append(ari_score)
    values = np.array(values)
    return np.argmax(values) + 1, values

def get_p_histo(X, K, y_test, seuil = 1e-7, max_iter = 50, p_max = None):
    """
    Compute the metrics for different values of p for Mc2PCA.
    p is the number of retained components.
    """
    if p_max is None:
        p_max = X.shape[1] # Works for DataFrame and numpy array
    idx_test = [np.where(y_test == str(i))[0] for i in range(1,np.max(y_test.astype(int))+1)]
    y_estimate = None
    precisions = []
    recalls = []
    aris = []
    for p in range(1, p_max + 1):
        model = Mc2PCA(K, p, seuil, max_iter=max_iter)
        idx_estimate, _, _ = model.fit(X)
        y_estimate = np.arange(len(y_test))
        for i in range(len(idx_estimate)):
            y_estimate[idx_estimate[i]] = i + 1
        precisions.append(compute_precision(idx_estimate, idx_test))
        aris.append(adjusted_rand_score(y_test.astype(int), y_estimate))
        recalls.append(compute_recall(idx_estimate, idx_test))
    return precisions, aris, recalls

def get_results_distance(X, K, y_test, seuil = 1e-7, max_iter = 50):
    """
    Compute the results for different distance metrics used for the reconstruction error.
    """
    idx_test = [np.where(y_test == str(i))[0] for i in range(1,np.max(y_test.astype(int))+1)]
    y_estimate = None
    precisions = []
    recalls = []
    aris = []
    distances = ["euclidean", "dtw", "l1", "cosine"]
    for distance in distances:
        model = Mc2PCA(K, 3, seuil, max_iter=max_iter, distance_metric=distance)
        idx_estimate, _, _ = model.fit(X)
        y_estimate = np.arange(len(y_test))
        for i in range(len(idx_estimate)):
            y_estimate[idx_estimate[i]] = i + 1
        precisions.append(compute_precision(idx_estimate, idx_test))
        aris.append(adjusted_rand_score(y_test.astype(int), y_estimate))
        recalls.append(compute_recall(idx_estimate, idx_test))
    return precisions, aris, recalls

def plot_info(X, K, seuil = 1e-7, max_iter = 50):
    """ 
    Plot the cumulative information for each cluster for different p.
    """
    nbre_features = X.shape[1] # Works for DataFrame and numpy array
    info_tot = []
    for p in range(1, nbre_features):
        model = Mc2PCA(K, p, seuil, max_iter=max_iter)
        _, _, info_by_cluster = model.fit(X)
        info_tot.append(info_by_cluster)
    plt.figure(figsize=(10, 7))
    info_tot.append([1 for y in range(K)])
    info_tot = np.array(info_tot)
    for i in range(K):
        plt.plot(np.arange(1, nbre_features + 1), info_tot[:, i], label = "Cluster " + str(i+1), marker = 'x')
    plt.legend() 
    plt.title("Cumulative Information for each cluster for different p")
    plt.xlabel("Number of p for Mc2PCA")
    plt.ylabel("Information")
    plt.xticks(np.arange(1, nbre_features + 1), rotation=45)

    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot the confusion matrix.
    """
    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)

    #Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    plt.show()

def metrics(y_test, idx_estimate,plot=True):
    """
    Compute the ARI score, precision and recall according to the results of the clustering.
    Plot the results by default.
    """

    # Convert y_test to a list of numpy arrays as in idx_estimate
    idx_test = [np.where(y_test == str(i))[0] for i in range(1,np.max(y_test.astype(int))+1)]

    # Convert idx_estimate to a list of numpy arrays as in y_test
    y_estimate = np.arange(len(y_test))
    for i in range(len(idx_estimate)):
        y_estimate[idx_estimate[i]] = i + 1 

    # Compute the metrics
    ari_score = adjusted_rand_score(y_test.astype(int), y_estimate)
    precision = compute_precision(idx_estimate, idx_test)
    recall = compute_recall(idx_estimate, idx_test)

    if(plot):
        print("Adjusted Rand Index:", ari_score) 
        print("Precision:", precision)
        print("Recall:", recall)
        
        plot_confusion_matrix(y_test.astype(int), y_estimate, classes = np.arange(1, np.max(y_test.astype(int) + 1)))
        plt.show()
    return ari_score,precision,recall