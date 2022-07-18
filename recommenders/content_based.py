"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
movies_df = pd.read_csv('resources/data/recommender_dataset.csv')
movies_df = movies_df.drop('Unnamed: 0', axis=1)


# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.
def content_model(movie_list, top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    # Subset of the data
    movies = movies_df[:27000]
    # Convenient indexes to map between movie titles and indexes of the 'all_df' dataframe
    indices = pd.Series(movies['title'])
    # Instantiating and generating the count matrix
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                         max_features=5000, stop_words='english')

    # Produce a feature matrix, where each row corresponds to a movie,
    # with TF-IDF features as columns
    tf_enrich_matrix = tf.fit_transform(movies['concat'])

    cosine_sim = cosine_similarity(tf_enrich_matrix, tf_enrich_matrix)

    # Getting the index of the movie that matches the title
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]
    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]
    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending=False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending=False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending=False)
    # Getting the indexes of the 10 most similar movies
    listings = score_series_1.append(score_series_1). \
        append(score_series_3).append(score_series_2).sort_values(ascending=False)

    # Store movie names
    recommended_movies = []
    # Appending the names of movies
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes, [idx_1, idx_2, idx_3])
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies['title'])[i])
    return recommended_movies
