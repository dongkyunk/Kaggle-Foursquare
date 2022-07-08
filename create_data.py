import dask
import sys
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from pytorch_lightning.utilities.distributed import rank_zero_info
from model.four_square_model import FourSquareModel
from config import register_configs, Config
from utils.utils import *
from sklearn.feature_extraction.text import TfidfVectorizer


@hydra.main(config_path=None, config_name="config")
def create_data(cfg: Config) -> None:
    pl.seed_everything(cfg.trainer_cfg.seed)
    rank_zero_info(OmegaConf.to_yaml(cfg=cfg, resolve=True))
    pd.options.mode.chained_assignment = None 

    # Load model
    model = FourSquareModel.load_from_checkpoint(cfg.path_cfg.train_1_lm_model_path, cfg=cfg, num_of_classes=369987)
    # torch.save(model.transformer.state_dict(), '/home/dongkyun/Desktop/Other/Kaggle-Foursquare/transformer_new.pth')
    model.eval()

    # Load data
    df = pd.read_parquet(cfg.path_cfg.ml_train_parquet_path)
    df = reduce_memory(df)
    df = df.fillna('')
    # df = df.head(1000)
    df = df.reset_index(drop=True)
    
    # Infer language embedding
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    device = torch.device("cuda")
    model.to(device) ## model to GPU

    vectors = create_text_embd(df, model, tokenizer, device)
    # vectors = torch.randn(len(df), 128).to(device)
    # vectors = torch.load('/home/dongkyun/Desktop/Other/Kaggle-Foursquare/text_embd.pt')

    k = 20
    k = min(len(df), k)

    # Get neighbors
    dists, nears = get_geospatial_neightbors(k, df)
    topk_indices, cos_sims_word, cos_sims_dist = get_cos_sims(k, vectors, nears)

    # Create pair dataframe
    train_df = create_pair_df(df, k, topk_indices, cos_sims_word, cos_sims_dist, nears, dists)

    # Add basic features
    train_df = create_features(df, train_df)

    # Pre inference tfidf
    tfidf_d = {}
    for col in vec_columns:
        tfidf = TfidfVectorizer()
        tv_fit = tfidf.fit_transform(df[col].fillna('nan'))
        tfidf_d[col] = tv_fit

    id2index_d = dict(zip(df['id'].values, df.index))
    df = df.set_index('id')

    train_df = reduce_memory(train_df)

    # Add more features
    chunks = [train_df[i:i+10000] for i in range(0, len(train_df), 10000)]
    chunked_chunks = [chunks[i:i+100] for i in range(0, len(chunks), 100)]

    for i, chunks in tqdm(enumerate(chunked_chunks)):
        temp = []
        for chunk in tqdm(chunks):
            chunk = create_more_features(df, chunk, tfidf_d, id2index_d)
            temp.append(chunk)

        train_df = pd.concat(temp)
        train_df['match'] = train_df.apply(lambda x: 1 if x['point_of_interest_1'] == x['point_of_interest_2'] else 0, axis=1)
        # Save training data
        train_df.to_parquet(f"/home/dongkyun/Desktop/Other/Kaggle-Foursquare/data/train_0_pair_{i}.parquet", index=False)
    

if __name__ == "__main__":
    sys.argv.append(f'hydra.run.dir={Config.path_cfg.save_dir}')
    register_configs()
    create_data()

