# obtain and save part(see the code) of attentions and logits for later use
# $ python inference.py [name] [artifact_path] [tuning_type]
# example: $ python inference.py FT3 artifacts/finetuned_BERT:v0/misty-dust-2.pth FT

import argparse
import os

import torch
import transformers
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding

# 0. argparse
parser = argparse.ArgumentParser()
parser.add_argument("name", help='path to save attentions and logits')
parser.add_argument("artifact_path", help="provide path to the artifact to load")
parser.add_argument("tuning_type", help="provide which tuning method was used for the artifact. Should be one of 'FT', 'Adapter', 'LoRA'")
args = parser.parse_args()


# 1. load model and load weights from artifact

## load base model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2).to(device)

## function to check env (transformer library)
def is_adapter_transformer():
    """function to check whether current env is 'adapter-transformer' or 'transformer'"""
    if 'AdapterConfig' in dir(transformers):
        return True
    return False

## load the weights from provided artifact
if args.tuning_type == 'FT':
    assert not is_adapter_transformer()
    model.load_state_dict(torch.load(args.artifact_path, map_location=device))
elif args.tuning_type == 'Adapter':
    assert is_adapter_transformer()
    model.load_adapter(args.artifact_path, set_active=True)
elif args.tuning_type == 'LoRA':
    assert not is_adapter_transformer()
    model.load_state_dict(torch.load(args.artifact_path, map_location=device), strict=False)
else:
    raise argparse.ArgumentError


# 2. model inference

## load dataset and build dataloader with dynamic padding
kr3_tokenized = Dataset.load_from_disk('kr3_tokenized')
kr3_tokenized.set_format(type='torch')

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dloader = DataLoader(kr3_tokenized, batch_size = 32, collate_fn=data_collator) # Important not to shuffle.

## inference
model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(dloader)):
        # forward
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, output_attentions=True)

        # full attn weights
        # attentions.size() = [batch_size, num_layers, num_heads, seq_len, seq_len]
        attentions = torch.stack([attn.detach() for attn in outputs.attentions], dim=1).to(device='cpu')

        # process attn: mean over head, attn w.r.t. the first token([CLS]). I'd love to store all, but time and memory...
        # attentions.size() = [batch_size, num_layers, seq_len]; seq_len is the max_seq_len within the batch.
        attentions = attentions.mean(dim=2)[:,:,0,:]

        # logits
        # logits.size() = [batch_size, 2]
        logits = outputs.logits.detach().to(device='cpu')

        # save attentions and logits (one file per one batch)
        dir_name = f'outputs/{args.name}'
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
            os.makedirs(f'{dir_name}/attentions')
            os.makedirs(f'{dir_name}/logits')

        torch.save(attentions, f'{dir_name}/attentions/batch_{i}.pt')
        torch.save(logits, f'{dir_name}/logits/batch_{i}.pt')