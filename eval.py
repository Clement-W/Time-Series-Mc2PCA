from sklearn.metrics import confusion_matrix, adjusted_rand_score
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes):
    labels = np.zeros_like(y_pred)
    for i in range(max(y_pred) + 1):
        mask = (y_pred == i)
        if np.sum(mask) > 0:
            unique, counts = np.unique(y_true[mask], return_counts=True)
            labels[mask] = unique[np.argmax(counts)]

    conf_matrix = confusion_matrix(y_true, labels)

    conf_matrix = confusion_matrix(y_true, y_pred, labels=all_labels)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    plt.show()

def compute_precision(C, G):
    """
    Compute the precision of clustering as described in the provided formula.
    
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
        precision += (len(cj) / N) * (max_intersection / len(cj))
        
    return precision