# FRESH(Faithful Rationale Extraction from Saliency tHresholding) from Jain et al., 2020
# functions for implementation. Intended for import in other files.

from typing import Iterable, Literal, Tuple

import torch


# 1. Normalize after excluding special tokens
def exclude_cls_sep(attn : torch.Tensor) -> torch.Tensor:
    '''
    Zero out the special tokens [CLS] and [SEP]. Then normalize over the remaining tokens.
    Args:
        attn: attention scores per tokens
    Return:
        normalized attention scores whose size is same as `attn`.

    Note that the code below is a shortcut for very simple setting where [CLS]
    comes first and [SEP] comes last, and they are the only speical tokens. Code should be 
    different in different settings, e.g. [SEP] in the middle, prompt exists.
    '''
    attn[0] = 0.0 # idx 0 for [CLS]
    attn[torch.nonzero(attn)[-1]] = 0.0 # last nonzero idx for [SEP]
    attn /= attn.sum() # normalize over the remanings
    return attn


# 2. Calculate attention scores per word
def get_attn_per_word(attn : torch.Tensor, words : Iterable, tokenizer) -> torch.Tensor:
    '''
    Calculate attention scores per word.
    Args:
        attn: attention scores per tokens
        words: iterable of words
        tokenizer: tokenizer tokenizes word to tokens
    Return:
        attention scores per word whose type is torch.Tensor and length is same as `words`.
    '''

    attn_per_word = torch.zeros(len(words))

    # start index by 1
    # why not 0? Because 0 corresponds to [CLS]. cf) We don't have [SEP] in the middle of the input, so we do not consider that.
    i = 1

    for j, word in enumerate(words):
        num_tokens = len(tokenizer.tokenize(word)) # number of tokens in the word

        # sum over tokens to obtain attention per word
        attn_per_word[j] = attn[i:i+num_tokens].sum()
        i += num_tokens

    assert attn[i+1] == 0.0 # attn score after all the indexing should be zero (corresponding to [SEP] after normalization)

    return attn_per_word
    

# 3. Obtain rationales
def discretize_attn(attn : torch.Tensor, 
                    words : Iterable, 
                    strategy : Literal['continguous', 'Top-k'] = 'Top-k',
                    ratio : float = 0.2) -> Tuple[list]:
    '''
    Dicscretize soft attentions scores into hard rationales.
    Args:
        attn: attention score per words
        words: iterable of words:
        strategy: Strategy for discretization, used in Jain et al., 2020
        ratio: ratio of length of rationale to whole input.
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
                unrationale.append(word)

        return rationale, unrationale


    else: # strategy == 'continguous'
        rationale_len = int(len(attn) * ratio)
        best_score = 0.0

        for i in range(len(attn) - rationale_len):
            rationale_score = attn[i:i+rationale_len].sum()
            if rationale_score > best_score:
                best_score = rationale_score
                rationale = words[i:i+rationale_len]
                unrationale = words[:i] + words[i+rationale_len:]
        
        return rationale, unrationale
