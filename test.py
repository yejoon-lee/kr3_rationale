import torch
from torchmetrics.functional import f1_score
device = torch.device('cuda')

target = torch.tensor([0,1,0,1,0,1], device=device)
preds = torch.tensor([1,1,0,1,1,0], device=device)
print(target.device, preds.device)
print(f1_score(preds, target, num_classes=2, average='none')[0].item())
print(f1_score(preds, target, num_classes=2, average='none')[1].item())
print(f1_score(preds, target, num_classes=2, average='macro').item())