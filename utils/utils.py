from cgi import test
import gc
import math
import torch
import difflib
import Levenshtein
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from numba import jit


def get_n_sample_from_each_class_with_minimum_m(n, m, df, target_col, seed):
    size = df.groupby(target_col).size()
    target = df.groupby(target_col).sample(
        n, random_state=seed)
    has_more_than_n_samples = size >= m
    has_more_than_n_samples.index = target.index
    target = target[has_more_than_n_samples]
    return target


def create_text_embd(df, model, tokenizer, device):
    vectors = []

    # Split dataframe into chunks
    chunks = [df[i:i+128] for i in range(0, len(df), 128)]

    for chunk in tqdm(chunks):
        input_ids = []
        attention_masks = []

        for index, row in chunk.iterrows():
            text = [row['name'], row['address'], row['city'], row['state'],
                    row['zip'], row['country'], row['url'], row['phone'], row['categories']]
            text = [str(x) for x in text]
            text = ' '.join(text)
            input = tokenizer(
                text,
                truncation=True,
                max_length=128,
                padding='max_length',
                return_tensors="pt"
            )
            input_ids.append(input['input_ids'])
            attention_masks.append(input['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        with torch.no_grad():
            embd = model(input_ids.to(device), attention_masks.to(device))

        vectors.append(embd)

    vectors = torch.cat(vectors, 0)
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)
    return vectors


def get_geospatial_neightbors(k, df):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(df[['latitude', 'longitude']], df.index)
    dists, nears = knn.kneighbors(df[['latitude', 'longitude']])
    return dists, nears


def get_cos_sims(k, vectors, nears):
    topk_indices = []
    cos_sims_word = []
    cos_sims_dist = []

    for i in tqdm(range(math.ceil(vectors.shape[0]/1000))):
        end = min(1000*(i+1), vectors.shape[0])
        dist = torch.mm(vectors[1000*i:end], vectors.t())
        value, indices = torch.topk(dist, k, dim=-1)

        temp = []
        for j in range(1000*i, end):
            temp.append(dist[j-1000*i][nears[j]])
        temp = torch.stack(temp, 0)

        cos_sims_dist.append(temp)
        cos_sims_word.append(value)
        topk_indices.append(indices)

    topk_indices = torch.cat(topk_indices, 0).cpu().numpy()
    cos_sims_word = torch.cat(cos_sims_word, 0).cpu().numpy()
    cos_sims_dist = torch.cat(cos_sims_dist, 0).cpu().numpy()

    return topk_indices, cos_sims_word, cos_sims_dist


def create_pair_df(df, k, topk_indices, cos_sims_word, cos_sims_dist, nears, dists):
    test_df = []

    for i in tqdm(range(k)):
        # Add based on geospatial distance
        cur_df = df[['id']]
        cur_df['match_id'] = df['id'].values[nears[:, i]]
        cur_df['dist'] = dists[:, i]
        cur_df['cos_sim'] = cos_sims_dist[:, i]
        cur_df['geospatial_k'] = k-i
        if i > 10:
            cur_df = cur_df[cur_df['dist'] < 2]
        cur_df = cur_df.drop(['dist'], axis=1)
        test_df.append(cur_df)

        # Add based on word similarity
        cur_df = df[['id']]
        cur_df['match_id'] = df['id'].values[topk_indices[:, i]]
        cur_df['cos_sim'] = cos_sims_word[:, i]
        cur_df['cos_sim_k'] = k-i
        if i > 10:
            cur_df = cur_df[cur_df['cos_sim'] > 0.5]
        test_df.append(cur_df)

    test_df = pd.concat(test_df)
    test_df = test_df.fillna(0)
    test_df = test_df.drop_duplicates()
    # test_df = test_df[~test_df[['id', 'match_id']].apply(frozenset, axis=1).duplicated()]
    return test_df


# @jit(nopython=True, cache=True)
# def LCS(S, T):
#     dp = [[0] * (len(T) + 1) for _ in range(len(S) + 1)]
#     for i in range(len(S)):
#         for j in range(len(T)):
#             dp[i + 1][j + 1] = max(dp[i][j] + (S[i] == T[j]),
#                                    dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1])
#     return dp[len(S)][len(T)]


@jit(nopython=True, cache=True)
def LCS(S, T):
    dp = np.zeros((len(S) + 1, len(T) + 1), dtype=np.int32)
    for i in range(len(S)):
        for j in range(len(T)):
            cost = (int)(S[i] == T[j])
            v1 = dp[i, j] + cost
            v2 = dp[i + 1, j]
            v3 = dp[i, j + 1]
            v4 = dp[i + 1, j + 1]
            dp[i + 1, j + 1] = max((v1,v2,v3,v4))
    return dp[len(S)][len(T)]
    

def create_features(df, test_df):
    df = df[['id', 'longitude', 'latitude', 'point_of_interest']]
    test_df = test_df.merge(df, on='id')
    test_df = test_df.rename(columns={
                             "longitude": "longitude_1", 'latitude': 'latitude_1', 'point_of_interest': 'point_of_interest_1'})
    test_df = test_df.merge(df, left_on='match_id', right_on='id')
    test_df = test_df.rename(columns={"longitude": "longitude_2", 'latitude': 'latitude_2',
                             'id_x': 'id', 'point_of_interest': 'point_of_interest_2'})
    test_df = test_df.drop(['id_y'], axis=1)

    test_df['longitude_diff'] = abs(
        test_df['longitude_2'] - test_df['longitude_1'])
    test_df['latitude_diff'] = abs(
        test_df['latitude_2'] - test_df['latitude_1'])
    test_df['euc_dist'] = (test_df['longitude_diff'] ** 2 + test_df['latitude_diff'] ** 2) ** 0.5
    test_df['manhattan_dist'] = test_df['longitude_diff'] + \
        test_df['latitude_diff']
    return test_df


feat_columns = ['name', 'address', 'city',
                'state', 'zip', 'url',
                'phone', 'categories', 'country']
vec_columns = ['name', 'categories', 'address',
               'state', 'url', 'country']

def reduce_memory(df):
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            cmin = df[col].min()
            cmax = df[col].max()
            if str(col_type)[:3] == 'int':
                if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif cmin > np.iinfo(np.int64).min and cmax < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
                #     df[col] = df[col].astype(np.float16)
                # el
                if cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

def create_more_features(df, test_df, tfidf_d, id2index_d):
    for col in feat_columns:
        if col in vec_columns:
            tv_fit = tfidf_d[col]
            indexs = [id2index_d[i] for i in test_df['id']]
            match_indexs = [id2index_d[i] for i in test_df['match_id']]
            test_df[f'{col}_sim'] = np.array(tv_fit[indexs].multiply(
                tv_fit[match_indexs]).sum(axis=1)).ravel()

        col_values = df.loc[test_df['id']][col].values.astype(str)
        matcol_values = df.loc[test_df['match_id']][col].values.astype(str)

        geshs = []
        levens = []
        jaros = []
        lcss = []
        for s, match_s in zip(col_values, matcol_values):
            if s != 'nan' and match_s != 'nan':
                geshs.append(difflib.SequenceMatcher(None, s, match_s).ratio())
                levens.append(Levenshtein.distance(s, match_s))
                jaros.append(Levenshtein.jaro_winkler(s, match_s))
                lcss.append(LCS(str(s), str(match_s)))
            else:
                geshs.append(np.nan)
                levens.append(np.nan)
                jaros.append(np.nan)
                lcss.append(np.nan)

        test_df[f'{col}_gesh'] = geshs
        test_df[f'{col}_leven'] = levens
        test_df[f'{col}_jaro'] = jaros
        test_df[f'{col}_lcs'] = lcss

        if col not in ['phone', 'zip']:
            test_df[f'{col}_len'] = list(map(len, col_values))
            test_df[f'match_{col}_len'] = list(map(len, matcol_values))
            test_df[f'{col}_len_diff'] = np.abs(
                test_df[f'{col}_len'] - test_df[f'match_{col}_len'])
            test_df[f'{col}_nleven'] = test_df[f'{col}_leven'] / \
                test_df[[f'{col}_len', f'match_{col}_len']].max(axis=1)

            test_df[f'{col}_nlcsk'] = test_df[f'{col}_lcs'] / test_df[f'match_{col}_len']
            test_df[f'{col}_nlcs'] = test_df[f'{col}_lcs'] / test_df[f'{col}_len']

            test_df = test_df.drop(f'{col}_len', axis=1)
            test_df = test_df.drop(f'match_{col}_len', axis=1)

        gc.collect()
    
    return test_df




