#!/usr/bin/python
# -*- coding: utf-8 -*-
# Importing the required modules

import pandas as pd
import numpy as np
import time
import sys
import warnings
from collections import defaultdict
from operator import itemgetter

# To make sure warnings are filtered out
warnings.filterwarnings("ignore")

col_name = ['user_id', 'item_id', 'ratings', 'timestamp']
# Reading from input csv files and storing in data frames
df = pd.read_csv("./ratings.csv")
movies = pd.read_csv("./movies.csv")
df = pd.merge(df, movies, on='movieId')
avg_rating_df = pd.DataFrame(df.groupby('title')['rating'].mean())
avg_rating_df['no_of_ratings'] = df.groupby('title')['rating'].count()
avg_rating_df['title'] = avg_rating_df.index
um_rating = df.pivot_table(index='userId', columns='title',
                           values='rating')


# Recommend function to output movies according to correlation
# to the movies present in database
def recommend(userID, genre=None):
    rec_mov = pd.DataFrame()
    user_rating = []

    if genre is None:
        genres = defaultdict(int)
        for i in um_rating.columns:
            if um_rating[i][userID] >= 4:  # Picking rating >4 movies
                user_rating.extend([i])
                temp = df.loc[df['title'] == i].iloc[0][5].split("|")
                for t in temp:
                    genres[t] += 1

        genres = dict(
            sorted(genres.items(), key=itemgetter(1), reverse=True)[:5])

    else:
        for i in um_rating.columns:
            if um_rating[i][userID] >= 4:  # Picking rating >4 movies
                user_rating.extend([i])

        genres = {genre: 0}

    user_rating_final = []
    for movie in user_rating:  # picking movies with >100 ratings
        if avg_rating_df.loc[movie]['no_of_ratings'] > 100:
            user_rating_final.extend([movie])

    rec = pd.DataFrame()
    rec = []
    for i in user_rating_final:
        rating = um_rating[i]
        similar = um_rating.corrwith(rating)
        corr = pd.DataFrame(similar, columns=['correlation'])
        corr.dropna(inplace=True)
        for j in corr.index:
            # Correlation is taken as 0.95 and not watched
            if corr.loc[j]['correlation'] >= 0.95 and j not in rec and j \
               not in user_rating:
                rec.extend([j])
                r = {'movie': [j],
                     'correlation': [round(corr.loc[j]['correlation'], 2)]}
                record = pd.DataFrame(data=r)
                rec_mov = rec_mov.append(record, ignore_index=True)
                rec_mov = rec_mov.sort_values(by=['correlation'],
                                              ascending=False)

    return rec_mov['movie'], genres


def get_rec_item(userID, genre=None):
    rec, genres = recommend(userID, genre)

    rec_list = []
    for y in rec:
        genre_y = movies.loc[movies['title'] == y].iloc[0][2].split("|")
        # print(genre_y)
        matched_gen = 0
        for g in genre_y:
            if g in genres.keys():
                matched_gen += 1
        # print(matched_gen)
        if matched_gen == len(genre_y):
            matched_rat = avg_rating_df.loc[avg_rating_df['title']
                                            == y].iloc[0][0]
            matched__num_rat = avg_rating_df.loc[avg_rating_df['title']
                                                 == y].iloc[0][1]

            rec_list.append((str(y), matched_rat, matched__num_rat))

    rec_list = sorted(rec_list, key=lambda x: [-x[1], -x[2]])[0:10]
    final_list = []
    for z1 in rec_list:
        final_list.append(z1[0])

    return list(set(final_list))
