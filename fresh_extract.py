# extract and save rationale from each examples using attention
# Implementation of FRESH(Faithful Rationale Extraction from Saliency tHresholding) from Jain et al., 2020
# Uses multiprocessing

# $ python fresh_extract.py [name] [save_path] [--strategy] [--ratio] 
# example: $ python fresh_extract.py FT3 rationales/FT3

import argparse
from glob import glob
from multiprocessing import Pool
from typing import Iterable, Literal, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import BertTokenizer


# 0. argparse
parser = argparse.ArgumentParser()
parser.add_argument("name", help='path to load attentions')
parser.add_argument("save_path", help='path to save the rationales')
# to understand two args below, see discretize_attn()
parser.add_argument("--strategy", help='Strategy for discretization, used in Jain et al., 2020', default='Top-k')
parser.add_argument("--ratio", help='ratio of length of rationale to whole input.', default=0.2)
args = parser.parse_args()


# 1. load dataset and tokenizer to get words and tokenize for each example
kr3 = load_dataset("Wittgensteinian/KR3", name='kr3', split='train')
kr3 = kr3.remove_columns(['__index_level_0__'])
kr3_binary = kr3.filter(lambda example: example['Rating'] != 2)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# 2. extract rationales for every examples in the batch
def extract(batch_i : int, 
            strategy : Literal['continguous', 'Top-k'] = 'Top-k', 
            ratio : float = 0.2):
    '''Extract rationales from examples using given heuristic and ratio.
    Args:
        batch_i: batch index used to load the attention weights
        strategy: Strategy for discretization, used in Jain et al., 2020
        ratio: ratio of length of rationale to whole input.
    '''
    kr3_rationale = Dataset.from_dict({}) # empty Dataset to store rationales, unrationales, and rating

    # load attention scores
    attns = torch.load(f'outputs/{args.name}/attentions/batch_{batch_i}.pt') # attns.size() = [batch_size, num_layers, seq_len]
    attns = attns[:,-1,:] # only attn from the LAST LAYER; attns.size() = [batch_size, seq_len]

    for example_i in range(attns.size()[0]): 
        # index attn of single example
        attn = attns[example_i, :]

        # get words & rating corresponding to the attn from the dataset
        # note that we match attentions and words/rating by their order. No shuffle was done during inference.
        words = kr3_binary[batch_i*32 + example_i]['Review'].split(' ')
        rating = kr3_binary[batch_i*32 + example_i]['Rating']
        
        # extract rationale using functions below
        attn = exclude_cls_sep(attn)
        attn_per_word = get_attn_per_word(attn, words, tokenizer)
        rationale, unrationale = discretize_attn(attn_per_word, words, strategy=strategy, ratio=ratio)

        # add item in the dataset
        kr3_rationale = kr3_rationale.add_item({'Rating':rating, 'Rationale':' '.join(rationale), 'Unrationale':' '.join(unrationale)})
    
    kr3_rationale.save_to_disk(f'{args.save_path}/batch_{batch_i}') # save the dataset


# functions below all operate on attention for SINGLE EXAMPLE (not batch)
# 2-1. Normalize attention after excluding special tokens
def exclude_cls_sep(attn : torch.Tensor) -> torch.Tensor:
    '''
    Zero out the special tokens [CLS] and [SEP]. Then normalize over the remaining tokens.
    Args:
        attn: attention scores per tokens
    Return:
        normalized attention scores whose size is same as `attn`.

    Note that the code below is a shortcut for very simple setting where [CLS]
    comes first and [SEP] comes last, and they are the only speical tokens. Code should be 
    different in different settings, e.g. where [SEP] is in the middle or prompt exists.
    '''
    attn[0] = 0.0 # idx 0 is [CLS]
    attn[torch.nonzero(attn)[-1]] = 0.0 # last nonzero idx is [SEP]
    attn /= attn.sum() # normalize over the remanings
    return attn


# 2-2. Calculate attention scores per word
def get_attn_per_word(attn : torch.Tensor, words : Iterable, tokenizer) -> torch.Tensor:
    '''
    Calculate attention scores per word.
    Args:
        attn: attention scores per tokens
        words: iterable of words
        tokenizer: tokenizer tokenizes word to tokens
    Return:
        attention scores per word whose type is torch.Tensor and length is the same to `words`.
    '''

    attn_per_word = torch.zeros(len(words)) # storage

    # start index by 1
    # why not 0? Because 0 corresponds to [CLS]. cf) We don't have [SEP] in the middle of the input, so we do not consider that.
    i = 1

    for j, word in enumerate(words):
        num_tokens = len(tokenizer.tokenize(word)) # number of tokens in the word

        # sum over tokens within a word to obtain attention per word
        attn_per_word[j] = attn[i:i+num_tokens].sum()
        i += num_tokens

        if i > 511: # this implies #(tokens) > 512, hence truncated. Remaning words were not used in the original input. Just let attn=0 for truncated words.
            return attn_per_word

    assert attn[i] == 0.0 # attn score after all the indexing should be zero (corresponding to [SEP] after normalization)
    
    return attn_per_word
    

# 2-3. Obtain rationales according to strategy
def discretize_attn(attn : torch.Tensor, 
                    words : Iterable, 
                    strategy : Literal['continguous', 'Top-k'] = 'Top-k',
                    ratio : float = 0.2) -> Tuple[list]:
    '''
    Dicscretize soft attentions scores into hard rationales.
    Args:
        attn: attention score per words
        words: iterable of words
        strategy: Strategy for discretization, used in Jain et al., 2020
        ratio: the length of rationale / the length of whole input
    Return:
        Tuple of (rationale, unrationale)
        Rationale is the list of words for the rationales.
        Unrationale is the list of remaining words, not included in the rationale.
    '''

    assert len(attn) == len(words)

    if strategy == 'Top-k':
        rationale_idxs = attn.argsort(descending=True)[:int(len(attn)*ratio)]
        rationale = [] # words in the rationale, ordered
        unrationale = [] # words not in the rationale, ordered

        for i, word in enumerate(words):
            if i in rationale_idxs:
                rationale.append(word)
            else:
                if attn[i].is_nonzero(): # if attn per word is zero, it's likely that the word is just truncated.
                    unrationale.append(word)

        return rationale, unrationale


    else: # strategy == 'continguous'
        rationale_len = int(len(attn) * ratio)
        best_score = 0.0

        for i in range(len(attn) - rationale_len):
            rationale_score = attn[i:i+rationale_len].sum()
            if rationale_score > best_score: # best continguous rationale so far
                best_score = rationale_score
                rationale = words[i:i+rationale_len]
                
                # unrationale (exclude rationale & truncated words(those with attn == 0))
                unrationale_preceding = words[:i]

                idx = attn.nonzero().squeeze()[-1].item() # last nonzero index in attn
                unrationale_proceeding = words[i + rationale_len : idx + 1] 

                unrationale = unrationale_preceding + unrationale_proceeding

        return rationale, unrationale


# 3. execute extract() with multiprocessing
batch_i_list = [i for i in range(len(glob(f'outputs/{args.name}/attentions/batch_*.pt')))]

with Pool(128) as p:
    p.map(extract, batch_i_list)