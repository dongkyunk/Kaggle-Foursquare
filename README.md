# Kaggle-Foursquare

Here is my solution for Kaggle Foursquare Location Matching. 

This solution received a silver medal without any ensembling or complicated feature engineering.

The pipeline is is simple:

1. Train xlmroberta with ArcFace Loss
2. Use the cos sims from the xlmroberta + coordinate distance to extract match candidates
3. Add features (cos sim, distance, lcs, tfidf, etc…)
4. Train a lightgbm model (with flaml hyperparameter optimization) to select the correct candidates as a binary classification task.
5. Do 2-3 on the test data and inference with lgbm
