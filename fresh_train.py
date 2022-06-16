# train end-task with extracted rationales
# Implementation of FRESH(Faithful Rationale Extraction from Saliency tHresholding) from Jain et al., 2020

# $ python fresh_train.py [name]
# example: $ python fresh_train.py FT3_TopK

from argparse import ArgumentParser
from glob import glob

import torch
from datasets import load_from_disk
from torchmetrics import F1Score
from tqdm import tqdm
from transformers import BertForSequenceClassification

import wandb

# 0. Setup
# argparse
parser = ArgumentParser()
parser.add_argument("name", help="path to load tokenized rationales(KR4)")
parser.add_argument("--lr", help="learning rate", default=1e-5)
args = parser.parse_args()

# variables (not necessarily hyperparams)
MAX_EPOCH = 8
N_BATCH_LOG = 2000


# 1. Prepare training
# split dataset (8 : 2). w/o validation to follow the setup of KR3
paths = glob(f'kr4_tokenized/{args.name}/batch_*') # behavior of glob is deterministic.
train_paths = paths[ : int(len(paths)*.8)]
test_paths = paths[int(len(paths)*.8) : ]

# device, model, optimizer
device = torch.device('cuda')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=5e-6)

# metric(F1) using torchmetrics
train_f1 = F1Score(num_classes=2, average='macro')
test_f1 = F1Score(num_classes=2, average='macro')


# 2. Train and test
# set up wandb
wandb.init(name=args.name, project='kr4')

# set initial values
batch_idx = 0
running_loss_batch = 0.0 # loss for every N_BATCH_LOG batches
best_f1 = 0.0
skipped_batches = 0


# train and test loop
for epoch in range(MAX_EPOCH):
    # train
    running_loss_epoch = 0.0
    model.train()
    for path in tqdm(train_paths):
        batch_idx += 1

        # load KR4
        batch = load_from_disk(path)

        # skip too long seq, due to out-of-memory
        if len(batch['input_ids'][0]) > 480:
            skipped_batches += 1
            continue

        # forward 
        tensor_batch = {k:torch.tensor(v).to(device) for k,v in batch.to_dict().items()}
        out = model(**tensor_batch)
        loss = out.loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update loss
        running_loss_batch += loss.item()
        running_loss_epoch += loss.item()

        # update f1
        target = torch.tensor(batch['labels'])
        preds = out.logits.argmax(axis=1).cpu()
        train_f1.update(preds, target)
        
        # log loss per batch
        if batch_idx % N_BATCH_LOG == 0:
            wandb.log({'batch':batch_idx, 'loss':running_loss_batch / N_BATCH_LOG})
            running_loss_batch = 0.0

    # test
    model.eval()
    with torch.no_grad():
        test_running_loss = 0.0
        for path in tqdm(test_paths):
            # load
            batch = load_from_disk(path)

            # forward
            tensor_batch = {k:torch.tensor(v).to(device) for k,v in batch.to_dict().items()}
            out = model(**tensor_batch)
            loss = out.loss

            # update loss
            test_running_loss += loss.item()

            # update f1
            target = torch.tensor(batch['labels'])
            preds = out.logits.argmax(axis=1).cpu()
            test_f1.update(preds, target)

    # compute f1 and reset Metric
    total_train_f1 = train_f1.compute().item()
    total_test_f1 = test_f1.compute().item()
    train_f1.reset()
    test_f1.reset()

    # log history (per epoch)
    wandb.log({'epoch':epoch+1,
                'train_loss':running_loss_epoch / len(train_paths),
                'test_loss':test_running_loss / len(test_paths),
                'train_f1':total_train_f1,
                'test_f1':total_test_f1,
                'skipped_batches':skipped_batches,
                })

    # log summary
    if total_test_f1 > best_f1:
        best_f1 = total_test_f1
        wandb.run.summary['best_f1'] = best_f1