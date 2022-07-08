import pandas as pd
from flaml import AutoML
from sklearn.model_selection import GroupKFold


df = [pd.read_parquet("/home/dongkyun/Desktop/Other/Kaggle-Foursquare/data/train_1_pair.parquet")]
for i in range(14):
    df.append(pd.read_parquet(f"/home/dongkyun/Desktop/Other/Kaggle-Foursquare/data/train_0_pair_{i}.parquet"))
df = pd.concat(df)
# df = df[df[['id', 'match_id']].apply(frozenset, axis=1).duplicated()] 
# df = df.head(1)
# df.to_csv('test.csv', index=False)
df = df[~df[['id', 'match_id']].apply(frozenset, axis=1).duplicated()] 
df = df[df['id']!=df['match_id']]
df['euc_dist'] = (df['longitude_diff'] **
                        2 + df['latitude_diff'] ** 2) ** 0.5




# print(df.columns)
# # Count cases with match false with euc dist lower than 5
# df = df[df['manhattan_dist'] > 200]
# df['exceptions'] = df.apply(lambda row: 1 if row['match'] else 0, axis=1)
# print(df.head())
# print(df['exceptions'].head())
# print(df['exceptions'].sum())
# print(df['exceptions'].mean())

# kfold = GroupKFold(n_splits=2)

# for n, (_, test_index) in enumerate(kfold.split(df, groups=df['id'])):
#     df.loc[test_index, 'kfold'] = int(n)

# df_0 = df[df['kfold'] == 0]
# df_1 = df[df['kfold'] == 1]

df = df.reset_index(drop=True)
X_train = df.drop(['match', 'id', 'match_id', 'point_of_interest_1', 'point_of_interest_2'], axis=1)
y_train = df['match'].astype(int)
print(y_train.value_counts())
del df
# X_test = df_1.drop(['match', 'id', 'match_id', 'point_of_interest_1', 'point_of_interest_2'], axis=1)
# y_test = df_1['match'].astype(int)

# print(X_train.max(axis = 0))

# print(y_train.head())

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
# automl = AutoML(use_ray=False, time_budget=2400, early_stop=True)
# automl.fit(X_train, y_train, task="classification", estimator_list=["xgboost"])#,"xgboost"])
skf = StratifiedKFold(n_splits=3, random_state=42)
for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
    model = LGBMClassifier(colsample_bytree=0.6206704891330953,
                learning_rate=0.017986732071296262, max_bin=1023,
                min_child_samples=9, n_estimators=6118, num_leaves=298,
                reg_alpha=0.00706357318094864, reg_lambda=0.05558721086102777,
                verbose=-1)
    model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
    model.booster_.save_model(f'lgbm_20_{i}.txt')


# model = LGBMClassifier(colsample_bytree=0.6206704891330953,
#             learning_rate=0.017986732071296262, max_bin=1023,
#             min_child_samples=9, n_estimators=6118, num_leaves=298,
#             reg_alpha=0.00706357318094864, reg_lambda=0.05558721086102777,
#             verbose=-1)
# model.fit(X_train, y_train)
# model.booster_.save_model('lgbm_20_other.txt')

# # # model = pickle.load(open("/home/dongkyun/Desktop/Other/Kaggle-Foursquare/data/automl.pkl", "rb"))

# model = XGBClassifier(base_score=0.5, booster='gbtree',
#               colsample_bylevel=0.2722707486171394, colsample_bynode=1,
#               colsample_bytree=0.6588510341925055, gamma=0, gpu_id=-1,
#               grow_policy='lossguide', importance_type='gain',
#               interaction_constraints='', learning_rate=0.152644192676315,
#               max_delta_step=0, max_depth=0, max_leaves=270,
#               min_child_weight=1.3257443894173804, missing=None,
#               monotone_constraints='()', n_estimators=1374, n_jobs=-1,
#               num_parallel_tree=1, random_state=0,
#               reg_alpha=0.0015140201648039238, reg_lambda=0.0653282736317393,
#               scale_pos_weight=1, subsample=0.9432092150938152,
#               tree_method='hist', use_label_encoder=False,
#               validate_parameters=1, verbosity=0)
# model.fit(X_train, y_train)
# model.get_booster().save_model('xgb_20.model')

# # y_pred = model.predict(X_test)

# # # Calculate the accuracy
# # from sklearn.metrics import accuracy_score
# # print(accuracy_score(y_test, y_pred))

# pickle.dump(model, open("/home/dongkyun/Desktop/Other/Kaggle-Foursquare/data/lgbm2.pkl", "wb"))

# import treelite
# model = pickle.load(open("/home/dongkyun/Desktop/Other/Kaggle-Foursquare/data/lgbm2.pkl", 'rb'))

# model = treelite.Model.load(f'model_fold{fold}.txt', model_format='lightgbm')

