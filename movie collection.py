import pandas as pd
collection = pd.read_csv("movie_collection_test.csv", index_col=0)
print(collection.columns)

# Well, you have successfully treated data
# and it is ready to be used to train the model.
#
# Here's your eighth task of the project: Find the relationship
# between budget of the movie and the amount
# it will gross at the box office (collection).

