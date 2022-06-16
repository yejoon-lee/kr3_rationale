import wandb
run = wandb.init()
artifact = run.use_artifact('wittgensteinian/Parameter-Efficient-Tuning/Adapter_BERT:v1', type='model')
artifact_dir = artifact.download()