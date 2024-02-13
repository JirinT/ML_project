import pandas as pd

df = pd.read_csv("caxton_dataset_final.csv")

df_size = df.shape
labels = df.columns

print(labels.to_list())