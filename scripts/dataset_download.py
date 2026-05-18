import pandas as pd
url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
df = pd.read_csv(url)
df.to_csv("Python/delaney_processed.csv", index=False)
print("Download complete")