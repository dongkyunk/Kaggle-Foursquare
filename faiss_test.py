import pandas as pd
import torch
from utils.utils import *
from tqdm import tqdm
pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_parquet('/home/dongkyun/Desktop/Other/Kaggle-Foursquare/data/train_1_pair.parquet')
df = df[df['id']!=df['match_id']]
print(df)

all_pair_df = pd.read_parquet('/home/dongkyun/Desktop/Other/Kaggle-Foursquare/all_train_1_pair.parquet')
all_pair_df = all_pair_df[all_pair_df['id']!=all_pair_df['match_id']]
print(all_pair_df)

# Compare all_pair_df and df based on columns id and match_id
a = df[['id', 'match_id']].values.tolist()
b = all_pair_df[['id', 'match_id']].values.tolist()
a = [x+''+y for x, y in a]
b = [x+''+y for x, y in b]
print(len(set(a) & set(b)))
print(len(set(a) & set(b))/len(set(b)))


# df2 = pd.read_parquet('/home/dongkyun/Desktop/Other/Kaggle-Foursquare/data/train_1.parquet')
# grouped = df2.groupby('point_of_interest')

# all_pair_df = []
# for index, row in tqdm(df2.iterrows()):
#     x = grouped.get_group(row['point_of_interest'])
#     cur_df = x[['id']]
#     cur_df.rename(columns={'id': 'match_id'}, inplace=True)
#     cur_df['id'] = row['id']
#     all_pair_df.append(cur_df)

# all_pair_df = pd.concat(all_pair_df)
# all_pair_df = all_pair_df.drop_duplicates()
# all_pair_df.to_parquet('all_train_1_pair.parquet', index=False)
# print(all_pair_df)

# vectors = torch.load('/home/dongkyun/Desktop/Other/Kaggle-Foursquare/text_embd.pt')


# train_df = pd.read_csv('data/train.csv')
# print(train_df.groupby('point_of_interest').count().max())
# print(len(train_df))

# # kfold = GroupKFold(n_splits=2)

# # for n, (_, test_index) in enumerate(kfold.split(train_df, groups=train_df['point_of_interest'])):
# #     train_df.loc[test_index, 'kfold'] = int(n)

# # train_df_0 = train_df[train_df['kfold'] == 0]
# # train_df_1 = train_df[train_df['kfold'] == 1]

# train_df.to_parquet('data/train.parquet')
# # train_df_1.to_parquet('data/train_1.parquet')

# # train_df_0 = pd.read_parquet('data/train_0.parquet')
# # train_df_1 = pd.read_parquet('data/train_1.parquet')

# # print(len(train_df_0))
# # print(len(train_df_1))

# # print(train_df_0['point_of_interest'].head())
# # print(train_df_1['point_of_interest'].head())