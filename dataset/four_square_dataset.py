import torch
from torch.utils.data import Dataset


class FourSquareDataset(Dataset):
    def __init__(self, cfg, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = cfg.trainer_cfg.max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        text = [sample['name'], sample['address'], sample['city'], sample['state'],
                        sample['zip'], sample['country'], sample['url'], sample['phone'], sample['categories']]
        text = [str(x) for x in text]
        text = ' '.join(text)
        
        inputs = self.tokenizer(text, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")

        return {
            'ids': inputs['input_ids'],
            'mask': inputs['attention_mask'],
            'latitude': torch.tensor(sample.latitude, dtype=torch.float),
            'longitude': torch.tensor(sample.longitude, dtype=torch.float),
            'label': torch.tensor(sample.point_of_interest, dtype=torch.long)
        }
