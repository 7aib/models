import pandas as pd
import numpy as np

path_to_file = "crude_oil_data.csv"

# Load the dataset and explore it
df = pd.read_csv(path_to_file)

print(df.head() )