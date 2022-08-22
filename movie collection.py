import pandas as pd
collection = pd.read_csv("movie_collection.csv", index_col=0)
print(collection.columns)


