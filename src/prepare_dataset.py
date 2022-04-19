import pandas as pd
import regex as re
from pyarabic import pyarabic.araby
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

class prepare_dataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
def clean_data(tweet):
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    clean_tweet = text.replace('@USER', '')
    clean_tweet = clean_tweet.replace('URL', '')
    clean_tweet=bytes(clean_tweet, 'utf-8').decode('utf-8','ignore')
    clean_tweet = re.sub(r'[A-Z]+',' ', clean_tweet)
    clean_tweet = clean_tweet.replace('>', '')
    clean_tweet = clean_tweet.replace('<', '')
    return clean_tweet

def get_max_len(self, tweets):
    tokenizer = self.tokenizer
    max_len = 0
    for tweet in tweets:
        tokens = tokenizer.encode(tweet, add_special_tokens = True)
        max_len = max(max_len, len(tokens))
    return max_len

def encode(self, tweets, tokenizer, max_len):
    
    tokenizer = self.tokenizer
    input_ids = []
    attention_masks = []
    for tweet in tweets:
        encodings_dict = tokenizer.encode_plus(
        tweet,
        add_special_tokens = True,
        max_length = max_len,
        pad_to_max_length = True,
        return_tensors = 'pt'
    )
        input_ids.append(encodings_dict['input_ids'])
        attention_masks.append(encodings_dict['attention_mask'])

  
    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    
    return input_ids, attention_masks


def create_dataloaders(input_ids, attention_masks,
                       labels, batch_size, dataset= None):
    
    tensor_dataset = TensorDataset(input_ids, attention_masks, 
                                   labels)

    dataloader = DataLoader(
        tensor_dataset,
        shuffle = False,
        batch_size = batch_size
    )
    
    return dataloader
    
