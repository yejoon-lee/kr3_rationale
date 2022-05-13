# FRESH(Faithful Rationale Extraction from Saliency tHresholding) from Jain et al., 2020
# Inference on batch included


import argparse
from typing import Literal
from datasets import Dataset
from glob import glob

import torch

from fresh_func import discretize_attn, exclude_cls_sep, get_attn_per_word

# 0. argparse
parser = argparse.ArgumentParser()
parser.add_argument("name", help='path to load attentions and logits')
args = parser.parse_args()


def fresh(tokenizer,
          strategy : Literal['continguous', 'Top-k'] = 'Top-k', 
          ratio : float = 0.2):
    
    kr3_rationale = Dataset.from_dict({}) # empty Dataset to store rationales and ratings

    for batch_i in range(len(glob(f'outputs/{args.name}/attentions/batch_*.pt'))):
        attns = torch.load(f'outputs/{args.name}/attentions/batch_{batch_i}.pt')
        for attn in attns[:,-1,:]:
            # from fresh_func
            attn = exclude_cls_sep(attn)
            attn_per_word = get_attn_per_word(attn, words, tokenizer)
            rationale, unrationale = discretize_attn(attn_per_word, words, strategy=strategy, ratio=ratio)

            # add item in the dataset
            kr3_rationale.add_item({'Rating':rating, 'Rationale':' '.join(rationale)})
            

