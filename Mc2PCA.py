import numpy as np
import pandas as pd
from tqdm import tqdm

def Mc2PCA(df, K, p, epsilon, max_iter= 100):
    """
    Perform the Mc2PCA algorithm on the given DataFrame.
    Implementation following the algorithm described in the paper:
    Li, H. (2019). Multivariate time series clustering based on common principal component analysis. Neurocomputing, 349.

    Args: 
        df (DataFrame): The input DataFrame containing the multivariate time series data with samples as rows and variables as columns, and each cell containing a pandas Series object or a numpy ndarray.
        K (int): The number of clusters to form using k-means.
        p (int): The number of principal components to retain in CPCA.
        epsilon (float): The threshold for convergence. 
        max_iter (int, optional): The maximum number of iterations for the clustering algorithm. Defaults to 100.

    Returns:
        tuple: A tuple containing two elements:
            - list: A list containing K arrays, each array containing the indices of the samples in the kth cluster.
            - list: The list of errors at each iteration.
    """

    # Center the data
    df = center_data(df)
    # Compute the covariance matrices of each time series
    cov_matrices = compute_covariance_matrices(df)
 
    # Initialize the indices
    idx = np.array_split(np.arange(df.shape[0]), K)
    # Initialize the associated common spaces
    S = compute_common_spaces(cov_matrices,idx,p)

    # Store the errors
    E = [np.inf]

    for t in tqdm(range(1, max_iter + 1)):

        # Assign the clusters based on k-means
        I,v = assign_clusters(df,S,K)
        E.append(np.sum(v)/len(v)) # normalize the error

        # Check convergence
        if np.abs(E[t-1] - E[t]) < epsilon:
            break
        
        # Assign new clusters
        idx = [np.where(I == k)[0] for k in range(K)]

        # Compute the new common spaces after the assignment
        S = compute_common_spaces(cov_matrices,idx,p)

    return idx, E


def center_data(X) :
    """
    Center the data by subtracting the mean of each time series from each cell.

    Args:
        X (DataFrame): The input DataFrame containing the multivariate time series data with samples as rows and variables as columns, and each cell containing a pandas Series object or a numpy ndarray.
    
    Returns:
        DataFrame: The centered DataFrame.
    """
    # Compute the means of each time series 
    means = X.map(lambda series: np.mean(series) if series is not None else np.nan)
    # Subtract the means from each cell
    normalized_X = (X - means).copy()

    # If the cells contains a pandas Series object, convert it into a numpy array
    if isinstance(normalized_X.iloc[0,0], pd.Series):
        for col_name in X.columns:
            normalized_X[col_name] = normalized_X[col_name].apply(lambda series: series.to_numpy() if series is not None else np.nan)

    return normalized_X

def compute_covariance_matrices(centered_df):
    """
    Compute the covariance matrix of each time series in the given DataFrame.

    Args:
        df (DataFrame): The input DataFrame containing the centered multivariate time series data.
    
    Returns:
        list: A list containing the covariance matrix of each time series in the given DataFrame.
    """

    # Store the covariance matrix of each time series
    cov_matrices = []

    for _, row in centered_df.iterrows():
        # Compute a matrix of size (n_variables, series_length) to represent the time series
        row_data = np.stack(row.values)
        # Compute the covariance matrix of the time series (n_variables, n_variables), suppose that df is centered
        cov_matrix = np.cov(row_data, bias=True)  
        # Add the covariance matrix to the list
        cov_matrices.append(cov_matrix)
    return cov_matrices

def CPCA(Sigma, p):
    """
    Perform Common Principal Component Analysis on a set of covariance matrices corresponding to
    a cluster of a multivariate time series, and return the common space of the cluster.

    Args:
        Sigma (list of ndarray): The list of covariance matrices.
        p (int): The number of principal components to retain.

    Returns:
        ndarray: The common space of the cluster.
    """
    mean_cov = np.mean(Sigma, axis=0) # mean covariance matrix of the cluster
    _, _, vt = np.linalg.svd(mean_cov) # SVD of the mean covariance matrix
    return vt[:p,:].T # return the first p principal components


def compute_common_spaces(cov_matrices,cluster_indices,p):
    """
    Compute the common principal components of each cluster using CPCA.

    This function iterates over the provided cluster indices and applies CPCA to the 
    covariance matrices corresponding to each cluster. This is used to find a common 
    subspace for each cluster that captures the most variance.

    Args:
        cov_matrices (list of ndarray): A list of covariance matrices for all samples in the dataset.
        cluster_indices (list of list of int): A list of K lists, each containing the indices of the samples in the kth cluster.
        p (int): The number of principal components to retain in CPCA.

    Returns:
        list of ndarray: A list of K arrays, each array containing the common space of the kth cluster.
    """
    S = []
    for indices in cluster_indices:
        if len(indices) > 0: # ensure that the cluster is not empty
            S.append(CPCA([cov_matrices[i] for i in indices], p))
        else:
            S.append(None)
    return S


def assign_clusters(df,S,K):
    """
    Assign each multivariate time series to a cluster based on the reconstruction error.

    Compute the reconstruction error for each time series after projecting it onto the common space of each cluster.
    Each time series is then assigned to the cluster for which it has the lowest reconstruction error.
    For empty clusters (very unlikely), assign a high error value.

    Args:
        df (DataFrame): The input DataFrame containing the centered multivariate time series data.
        S (list of ndarray): A list of K arrays, each array containing the common space of the kth cluster.
        K (int): The number of clusters.
    
    Returns:
        tuple: A tuple containing two elements:
            - ndarray: An array containing the indices of the clusters to which each time series is assigned.
            - ndarray: An array containing containing the minimum reconstruction error for each time series.
    """
    n = df.shape[0]
    Error = np.zeros((n,K))
    
    for k in range(K):
        if(S[k] is not None):
            # Compute the sum of squares transformation matrix
            sst = np.matmul(S[k], S[k].T)
            # Compute the reconstruction error for each time series
            for i in range(n):
                Y = np.matmul(df.iloc[i],sst)
                err = (np.linalg.norm(df.iloc[i] - Y))
                Error[i,k]= err.sum()/len(err) # normalize the error
        else:
            # Assign a high error value for empty clusters
            Error[:,k] = np.inf 

    # Assign each time series to the cluster with the lowest reconstruction error
    I = np.argmin(Error, axis=1)
    # Store the minimum reconstruction error for each time series
    v = Error[np.arange(n), I]
    return I,v