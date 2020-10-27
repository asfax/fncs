#!/usr/bin/python
# -*- coding: utf-8 -*-
# Importing the required modules
import pandas as pd
import numpy as np
import time
import re
import sys
import warnings
from collections import defaultdict
from operator import itemgetter
# To make sure warnings are filtered out
warnings.filterwarnings("ignore")
col_name = ['user_id', 'item_id', 'ratings', 'timestamp']
# Reading from input csv files and storing in data frames
df = pd.read_csv('./ratings.csv')
movies = pd.read_csv('./movies.csv')
df = pd.merge(df, movies, on='movieId')
avg_rating_df = pd.DataFrame(df.groupby('title')['rating'].mean())
avg_rating_df['no_of_ratings'] = df.groupby('title')['rating'].count()
avg_rating_df['title'] = avg_rating_df.index
avg_rating_df.head()
um_rating = df.pivot_table(index='title', columns='userId',
                           values='rating')
rec_mov = pd.DataFrame()

# Recommend function to output movies according to correlation
# to the movies present in database


def recommend(userID, genre=None):
    # print(um_rating.index)
    user_movies = []

    if not genre:
        genre_movie = defaultdict(int)
        for k in range(len(um_rating[userID])):
            if str(um_rating[userID].iloc[k]) != 'nan':
                user_movies.append(um_rating[userID].index[k])
                temp = df.loc[df['title'] ==
                              um_rating[userID].index[k]].iloc[0][5].split("|")

                for i in temp:
                    genre_movie[i] += 1

        top_genre = dict(sorted(genre_movie.items(),
                                key=itemgetter(1), reverse=True)[:5])
    else:
        for k in range(len(um_rating[userID])):
            if str(um_rating[userID].iloc[k]) != 'nan':
                user_movies.append(um_rating[userID].index[k])

        top_genre = {genre: 0}

    rec = pd.DataFrame()
    rec = []
    rating = um_rating[userID]
    similar = um_rating.corrwith(rating)
    corr = pd.DataFrame(similar, columns=['correlation'])
    corr.dropna(inplace=True)
    movie_list = []
    for j in corr.index:
      # Correlation is taken as 0.95 and not watched
        if corr.loc[j]['correlation'] >= 0.95 and j not in rec:
            rec.extend([j])
    for m in rec:
        for k in range(len(um_rating[m])):
            if um_rating[m].iloc[k] > 4 and um_rating[m].index[k] \
                    not in user_movies:  # Picking the movies that have similar watch history and rating >4
                movie_list.append(um_rating[m].index[k])
    return set(movie_list), top_genre


def get_rec_user(user_id, genre=None):
    rec, top_genre = recommend(user_id, genre)
    movies_recommend = []
    for z in rec:
        matched = 0
        genres_for_z = movies.loc[movies['title'] == z].iloc[0][2].split("|")
        for g in genres_for_z:
            if g in top_genre.keys():
                matched += 1

        if matched == len(genres_for_z):
            rat = avg_rating_df.loc[avg_rating_df['title'] == z].iloc[0][0]
            num_rat = avg_rating_df.loc[avg_rating_df['title'] == z].iloc[0][1]
            movies_recommend.append([z, rat, num_rat])

    rec_list = sorted(movies_recommend, key=lambda x: [-x[1], -x[2]])[:10]
    final_list = []
    for z1 in rec_list:
        final_list.append(z1[0])

    return final_list
