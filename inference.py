# obtain full attentions (over all layers and heads) and logits for later use
# $ python inference.py [name] [artifact_path] [tuning_type]
# example: $ python inference.py FT3 artifacts/finetuned_BERT:v0/misty-dust-2.pth FT

import argparse
import os

import torch
import transformers
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification

# 0. basic setup
parser = argparse.ArgumentParser()
parser.add_argument("name", help='path to save attentions and logits')
parser.add_argument("artifact_path", help="provide path to the artifact to load")
parser.add_argument("tuning_type", help="provide which tuning method was used for the artifact. Should be one of 'FT', 'Adapter', 'LoRA'")
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# 1. load model and load weights from artifact

## load base model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2).to(device)

def is_adapter_transformer():
    """
    function to check whether current env is 'adapter-transformer' or 'transformer'
    """
    if 'AdapterConfig' in dir(transformers):
        return True
    return False

## load the wei`ghts from provided artifact
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

## make
kr3_tokenized = Dataset.load_from_disk('kr3_tokenized_maxpad')
kr3_tokenized.set_format(type='torch') # kr3_tokenized_maxpad should have torch format, but somehow not. hgf bug??
dloader = DataLoader(kr3_tokenized, batch_size=64)

## empty tensors to store data
full_attentions = torch.Tensor(device='cpu')
full_logits = torch.Tensor(device='cpu')

## function to save attentions and logits
def save_tensors():
    dir_name = f'outputs/{args.name}'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    torch.save(full_attentions, f'outputs/{args.name}/attentions.pt')
    torch.save(full_logits, f'outputs/{args.name}/logits.pt')

## inference
model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(dloader)):
        # forward
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, output_attentions=True)

        # attn weights
        # attentions.size() = [batch_size, num_layers, num_heads, 512, 512]; seq_len padded to max(512).
        attentions = torch.stack([attn.detach() for attn in outputs.attentions], dim=1).to(device='cpu') 
        # mean over head, attn w.r.t. the first token([CLS]). I'd love to store all, but OOM.
        attentions = attentions.mean(dim=2)[:,:,0,:] 
        full_attentions = torch.cat((full_attentions, attentions), dim=0)

        # logits
        # logits.size() = [batch_size, 2]
        logits = outputs.logits.detach().to(device='cpu')
        full_logits = torch.cat((full_logits, logits), dim=0)

        # save
        checkpoint = 1000
        if i % checkpoint == 0:
            save_tensors()

save_tensors()