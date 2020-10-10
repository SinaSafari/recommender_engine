"""
Content Base Recommender Engine 

based on dataset we have, this engine recommend 10 similar movies to
the movie title that the user provide to the app
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# helpers
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


# path file
file_path = os.path.join(os.getcwd(), 'projects',
                         'movie_recommender_engine', 'movie_dataset.csv')


# dataframe
df = pd.read_csv(file_path)

# determine the features we want
features = ["keywords", "cast", "genres", "director"]

# new column in dataframe contains combinations of features
for feature in features:
    # fill the null values with empty string
    df[feature] = df[feature].fillna('')


# combine the features in to one string
def combine_features(row):
    try:
        # return f"{row["keyword"]} {row["cast"]} {row["genres"]} {row["director"]}"
        return row["keywords"] + " " + row["cast"] + " " + row["genres"] + " " + row["director"]
    except:
        print("Error: ", row)


# add new column to dataframe
df["combined_features"] = df.apply(combine_features, axis=1)

# create combine metrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)

# get movietitle from its title
movie_user_likes = input("please type the name of your favourite movie: ")

# titles in datasets are captalized
movie_index = get_index_from_title(movie_user_likes.title())
similar_movies = list(enumerate(cosine_sim[movie_index]))

# sort similar movies
sorted_similar_movies = sorted(
    similar_movies, key=lambda x: x[1], reverse=True)

# printing first 10 similar movies
i = 0
for elements in sorted_similar_movies:
    print(get_title_from_index(elements[0]))
    i = i + 1
    if i > 10:
        break
