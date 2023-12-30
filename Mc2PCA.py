import numpy as np
import pandas as pd
from tqdm import tqdm

# Fonctions utiles 
def normalize_series(series, column_mean):
    if series is not None:
        return series.to_numpy() - column_mean
    return series

def normalize(X) :
    means = X.map(lambda series: np.mean(series) if series is not None else np.nan)
    # Calcul de la moyenne par colonne des moyennes
    column_means = means.mean()

    # Appliquer la normalisation à chaque cellule
    normalized_X = pd.DataFrame()
    for col_name in X.columns:
        normalized_X[col_name] = X[col_name].apply(normalize_series, args=(column_means[col_name],))

    # Afficher les premières lignes des DataFrame normalisés pour vérifier
    return normalized_X

def cov_matrix(df):
    # suppose df is already normalized
    cov_matrices = []
    for index, row in df.iterrows():
        row_data = np.stack(row.values)

        cov_matrix = np.cov(row_data, bias=True)  

        cov_matrices.append(cov_matrix)
    return cov_matrices

# def CPCA(Sigma, p):
    # mean_cov = np.zeros_like(Sigma[0])
    # for cov in Sigma :
    #     mean_cov += cov
    # mean_cov /= len(Sigma)
#     u, _, _ = np.linalg.svd(mean_cov)
#     # u = np.asarray(u)
#     return u[:, :p]

def CPCA(Sigma, p):
    mean_cov = np.mean(Sigma, axis=0)
    u, _, _ = np.linalg.svd(mean_cov)
    return u[:, :p]


# On fait comme dans l'article, pas au nombre d'itérations mais on regarde l'évolution et on arrête en dessous d'un certain seuil
def Mc2PCA_old(df, K, p, seuil, max_iter= 100):
    # Normalize the data
    df = normalize(df)
    # Compute the covariance matrices
    cov_matrices = cov_matrix(df)
    n = df.shape[0]
    L = n//K
    extras = n % K
    start = 0
    indices = np.arange(n)
    idx = []
    S = []
    for k in range(0,K):
        end = start + L + (1 if k < extras else 0)
        idx.append(indices[start : end])
        start = end
        selected_cov = [cov_matrices[i] for i in idx[k]]
        S.append(CPCA(selected_cov, p))
    
    E = [np.inf]
    for t in tqdm(range(1, max_iter + 1)):
        Error = np.zeros((n,K))
        new_idx = [[] for _ in range(K)] #index
        sst = []
        for k in range(K):
            sst.append(np.matmul(S[k], S[k].T))
            for i in range(n):
                Y = np.matmul(df.iloc[i],sst[k])
                Error[i,k]= np.linalg.norm(np.linalg.norm(df.iloc[i] - Y))
        I = np.argmin(Error, axis=1)
        v = np.min(Error, axis=1)
        E.append(np.sum(v))
        for i in range(n):
            new_idx[I[i]].append(i)
        if np.abs(E[t-1] - E[t]) < seuil:
            break
        for k in range(0,K):
            selected_cov = [cov_matrices[i] for i in new_idx[k]]
            S[k] = CPCA(selected_cov, p)
    return new_idx, E


# On fait comme dans l'article, pas au nombre d'itérations mais on regarde l'évolution et on arrête en dessous d'un certain seuil
def Mc2PCA(df, K, p, seuil, max_iter= 100):
    # Normalize the data
    df = normalize(df)
    # Compute the covariance matrices
    cov_matrices = cov_matrix(df)
    n = df.shape[0]
    idx = np.array_split(np.arange(n), K)
    S = [CPCA([cov_matrices[i] for i in indices], p) for indices in idx]

    E = [np.inf]
    for t in tqdm(range(1, max_iter + 1)):
        Error = np.zeros((n,K))
        
        sst = []
        for k in range(K):
            sst.append(np.matmul(S[k], S[k].T))
            for i in range(n):
                Y = np.matmul(df.iloc[i],sst[k])
                Error[i,k]= (np.linalg.norm(df.iloc[i] - Y)).sum()

        I = np.argmin(Error, axis=1)
        v = Error[np.arange(n), I]
        E.append(np.sum(v))
        if np.abs(E[t-1] - E[t]) < seuil:
            break
        new_idx = [np.where(I == k)[0] for k in range(K)]
        S = [CPCA([cov_matrices[i] for i in indices], p) for indices in new_idx]

    return new_idx, E

