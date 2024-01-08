import pandas as pd 
import numpy as np
import os 
from scipy.io import arff


######## MOCAP ########
def parse_amc_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = {}
    for line in lines:
        if line.startswith(('#', ':', '!')) or line.strip().isdigit():
            continue

        parts = line.split()
        key = parts[0]
        values = [float(x) for x in parts[1:]]

        if key not in data:
            data[key] = []
        data[key].append(values)

    structured_data = {}
    for key, value_lists in data.items():
        for i, _ in enumerate(value_lists[0]):
            new_key = f"{key}_{i}"
            structured_data[new_key] = [values[i] for values in value_lists]

    for key in structured_data:
        structured_data[key] = np.array(structured_data[key])

    return structured_data

def extract_number(file_name):
    return int(file_name.split('.')[0].split('_')[1])


def load_MOCAP():
    path = './data/MOCAP/CMU_MOCAP_16/'
    paths = os.listdir(path)
    # Sort the paths by the number extracted
    sorted_paths = sorted(paths, key=extract_number)
    file_paths = [path + p for p in sorted_paths]

    # Store all the data in a DataFrame
    all_data = []
    for file_path in file_paths:
        file_data = parse_amc_file(file_path)
        all_data.append(file_data)

    X = pd.DataFrame(all_data)

    file_path = './data/MOCAP/cmu_meta.csv'
    df = pd.read_csv(file_path)
    df = df[df['Subject'] == 16].sort_values(by='path', key=lambda x: x.str.split('/').str[-1].str.split('_').str[-1].str.split('.').str[0].astype(int))
    y = df["walking_like"].to_numpy()
    y = np.array(['1' if i == False else '2' for i in y])
    return X, y

#####################

######### Epilepsy #########

def load_arff_data(file_path):
    # Load the data from the ARFF file
    data, meta = arff.loadarff(file_path)

    # Convert the data to a pandas DataFrame for easier manipulation
    df = pd.DataFrame(data)
    return df

def transform_dataframe(df, dim_number):
    """
    Transforme le DataFrame pour créer une colonne 'time_series_dimX' où X est le numéro de la dimension,
    et conserve la colonne 'activity'.
    """
    # Sélectionner toutes les colonnes sauf 'activity' pour créer la série temporelle
    time_series_columns = df.columns[:-1]  # Toutes les colonnes sauf la dernière

    # Créer une nouvelle colonne 'time_series_dimX' qui contient les valeurs des séries temporelles sous forme de numpy arrays
    column_name = f'dim{dim_number}'
    df[column_name] = df[time_series_columns].apply(lambda row: np.array(row), axis=1)
    
    # Créer un nouveau dataframe avec seulement la colonne 'time_series_dimX' et 'activity'
    new_df = df[[column_name, 'activity']]

    return new_df

def transform_data(paths_train):
    # Initialiser un DataFrame vide pour contenir les données transformées
    df_transformed = pd.DataFrame()

    # Boucle pour traiter chaque fichier
    for i, path in enumerate(paths_train):
        df = load_arff_data(path)
        df_dim_transformed = transform_dataframe(df, i)
        
        # Si c'est le premier fichier, initialiser df_transformed avec df_dim_transformed
        if df_transformed.empty:
            df_transformed = df_dim_transformed
        else:
            # Sinon, fusionner sur la colonne 'activity'
            df_transformed[f'dim{i}'] = df_dim_transformed[f'dim{i}']
    return df_transformed


def load_Epilepsy():
    path = './data/Epilepsy/Train/'
    paths = os.listdir('./data/Epilepsy/Train/')
    paths_train = [path + p for p in paths]
    df_transformed = transform_data(paths_train)
    y = df_transformed["activity"].to_numpy()
    X = df_transformed.drop("activity", axis=1)
    bib_convert = {b'EPILEPSY' : '1', b'RUNNING' : '2' , b'SAWING' : '3', b'WALKING' : '4'}
    y = np.array([bib_convert[i] for i in y])
    return X, y