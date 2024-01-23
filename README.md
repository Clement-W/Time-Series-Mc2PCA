# Time-Series-Mc2PCA

Analysis of the paper 
> Li, H. (2019). Multivariate time series clustering based on common principal component analysis. Neurocomputing, 349, 239-247

In this project, we focus on a thorough analysis and independent implementation from scratch of the method. We evaluated this method on a new dataset, along with two of the datasets originally used by the authors. Our exploration also extended to examining the method’s limitations and potential for improvement through four specific experiments: assessing performance with unbalanced data, evaluating sensitivity to noise, determining the optimal number of principal components to use, and experimenting with different distance metrics for calculating the reconstruction error. [Our report](TS_report.pdf) contains the paper analysis, our results and critical perspectives.

Files description:
- `Mc2PCA_class.py`: Contains the implementation of the Mc2PCA algorithm
- `load_data.py`: Load the datasets from the raw files
- `eval.py`: Contains the methods to compute evaluation metrics on the clustering results
- `experiments.ipynb`: Contains all the experiments performed

## Requirements
- Python >= 3.8

## Installation
To install the required dependencies, run the following command:
pip install -r requirements.txt

## Source of the data:

### CMU_MOCAP_S16
Carnegie Mellon University Motion Capture Database. CMU Motion Capture Database S16
Source: https://timeseriesclassification.com/description.php?Dataset=Epilepsy

### Japanese Vowels
Kudo,Mineichi, Toyama,Jun, and Shimbo,Masaru. Japanese Vowels. UCI Machine Learning Repository. https://doi.org/10.24432/C5NS47.
Loaded from the sktime library: https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.datasets.load_japanese_vowels.html

### Epilepsy
Villar, J., Vergara, P., Men'endez, M., Cal, E., Gonz'alez, V., & Sedano, J. (2016). Generalized models for the classification of abnormal movements in daily life and its applicability to epilepsy convulsion recognition. International journal of neural systems, 26(06), 1650037.
Source: https://timeseriesclassification.com/description.php?Dataset=Epilepsy
