I have this code
import pandas as pd
import scipy.stats
import os
import numpy as np

# Define the path to the folder containing the CSV files
folder_path = '../datasets/fastStorage/2013-8'

# Initialize a dictionary to store aggregated stats
aggregated_stats = {}

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Process each CSV file
for file in csv_files:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(folder_path, file), delimiter=';')

    df.columns = df.columns.str.strip()
    
    # Calculate statistics for each feature
    for column in df.columns:
        # Initialize the column in the dictionary if not already present
        if column not in aggregated_stats:
            aggregated_stats[column] = {"Mean": [], "Median": [], "Mode": [], "Standard Deviation": [], "Variance": [], 
                                        "Min": [], "Max": [], "Range": [], "25th Percentile": [], "50th Percentile": [], "75th Percentile": [],
                                        "Interquartile Range": [], "Skewness": [], "Kurtosis": [], "Sum": []}
        
        # Calculate and append statistics
        aggregated_stats[column]["Mean"].append(df[column].mean())
        aggregated_stats[column]["Median"].append(df[column].median())
        mode = df[column].mode()
        aggregated_stats[column]["Mode"].append(mode[0] if not mode.empty else np.nan)
        aggregated_stats[column]["Standard Deviation"].append(df[column].std())
        aggregated_stats[column]["Variance"].append(df[column].var())
        aggregated_stats[column]["Min"].append(df[column].min())
        aggregated_stats[column]["Max"].append(df[column].max())
        aggregated_stats[column]["Range"].append(df[column].max() - df[column].min())
        aggregated_stats[column]["25th Percentile"].append(df[column].quantile(0.25))
        aggregated_stats[column]["50th Percentile"].append(df[column].quantile(0.50))
        aggregated_stats[column]["75th Percentile"].append(df[column].quantile(0.75))
        aggregated_stats[column]["Interquartile Range"].append(df[column].quantile(0.75) - df[column].quantile(0.25))
        aggregated_stats[column]["Skewness"].append(scipy.stats.skew(df[column]))
        aggregated_stats[column]["Kurtosis"].append(scipy.stats.kurtosis(df[column]))
        aggregated_stats[column]["Sum"].append(df[column].sum())

# Convert the aggregated stats to a more friendly format (e.g., DataFrame) for reporting or further analysis
# Here's an example of converting the mean statistics to a DataFrame
all_stats_df = pd.DataFrame(aggregated_stats)


Then when running the next block

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

# ... (code for loading data from CSVs remains the same) ...

# Feature scaling (important for PCA, sometimes beneficial for t-SNE/UMAP)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(all_stats_df)

# Principal Component Analysis (PCA)
pca = PCA()
pca_components = pca.fit_transform(scaled_data)

I get the following error
TypeError: float() argument must be a string or a number, not 'list'

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
<ipython-input-8-681b7a000de1> in <module>
     10 from sklearn.preprocessing import StandardScaler
     11 scaler = StandardScaler()
---> 12 scaled_data = scaler.fit_transform(all_stats_df)
     13 
     14 # Principal Component Analysis (PCA)

How do I resolve?