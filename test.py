from sktime.datasets import load_japanese_vowels
from Mc2PCA import Mc2PCA
import matplotlib.pyplot as plt 

X, y = load_japanese_vowels(return_X_y=True)

K = 9
p = 3
seuil = 1e-7
new_idx, E = Mc2PCA(X, K, p, seuil, max_iter= 10)
plt.plot(E)