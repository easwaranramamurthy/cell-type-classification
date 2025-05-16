import torch
import torch.nn as nn

from tqdm import tqdm
import wandb
import numpy as np
import pandas as pd
import time

from models.transformer import Transformer
from datasets.npy_dataset import NpyDataset

def accuracy(preds:torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """computes accuracy for multi class classification

    Args:
        preds (torch.Tensor): model predictions
        labels (torch.Tensor): labels

    Returns:
        tuple[int, int]: number of correct preds, number of samples
    """
    pred_classes = preds.argmax(dim=1)
    correct = (pred_classes == labels).sum().item()
    total = labels.shape[0]
    return correct, total

def compute_metrics(net: nn.Module, dataset: NpyDataset, loss_fn: nn.Module, batch_size:int=64, num_workers:int=6) -> tuple[int, int]:
    """Do inference and compute loss and accuracy on a dataset

    Args:
        net (nn.Module): neural network
        dataset (NpyDataset): numpy dataset
        batch_size (int, optional): Batch size for inference. Defaults to 64.
        num_workers (int, optional): number of workers to use for data loader. Defaults to 6.
    Returns:
        tuple[int, int]: loss and accuracy
    """
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True, num_workers=num_workers)
    net.eval()
    loss = 0.0
    num_batches = 0
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for batch_x,batch_y in tqdm(dataloader, total=len(dataloader), position=3, leave=False, desc='inference'):
            batch_x = batch_x.to(device=device).to(torch.long)
            batch_y = batch_y.to(device=device)
            pred = net(batch_x)
            loss_value = loss_fn(pred,batch_y).item()

            correct, total = accuracy(pred, batch_y)
            num_correct += correct
            num_total += total
            loss+=loss_value
            num_batches+=1
    
    loss /= num_batches
    acc = num_correct / num_total
    
    return loss, acc

def train_model(net: nn.Module, train_dataset: NpyDataset, val_dataset: NpyDataset, epochs: int, batch_size_train: int=8, batch_size_infer: int=64, patience: int=5, num_workers: int=6) -> nn.Module:
    """Train the model to do multi class classification on an Npy dataset
    Args:
        net (nn.Module): model
        train_dataset (NpyDataset): training dataset
        val_dataset (NpyDataset): validation dataset
        epochs (int): number of epohcs
        batch_size_train (int, optional): Batch size for training. Defaults to 8.
        batch_size_infer (int, optional): Batch size for inference. Defaults to 64.
        patience (int, optional): number of epochs to wait until early stopping. Defaults to 5.
        num_workers (int, optional): number of workers to use for data loader. Defaults to 6.

    """
    timestamp = time.time()    
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size_train,shuffle=True, num_workers=num_workers)
    optim = torch.optim.AdamW(params = net.parameters(), lr=0.01)
    min_val_loss = float('inf')

    num_epochs_unsuccessful = 0

    # upweighting lowly represented classes
    _, counts = np.unique(train_dataset.y, return_counts=True)
    proportions = counts/ counts.sum()
    weights = torch.FloatTensor(1.0 / proportions).to(device='mps')

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    wandb.init(project="cell_type_classification",config={})
    print("Training model")
    for e in tqdm(range(epochs), position=0, desc='train_epochs'):
        net.train()
        for i, (batch_x, batch_y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), position=1, leave=False, desc='train_pass'):
            batch_x = batch_x.to(device=device).to(torch.long)
            batch_y = batch_y.to(device=device)
            optim.zero_grad()
            pred = net(batch_x)
            loss = loss_fn(pred,batch_y)
            loss.backward()
            optim.step()
        
        val_loss, val_acc = compute_metrics(net, val_dataset, loss_fn, batch_size_infer)
        train_loss, train_acc = compute_metrics(net, train_dataset, loss_fn, batch_size_infer)
        print({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        if val_loss < min_val_loss:
            num_epochs_unsuccessful = 0
            torch.save({'epoch': e,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss},
                        f'../checkpoints/model_{timestamp}.pth')
        else:
            num_epochs_unsuccessful+=1
        
        if num_epochs_unsuccessful>patience:
            print(f'Early stopping since no improvement in validation loss for {patience} epochs')
            break

    torch.save(net, f'../checkpoints/final_model_{timestamp}.pth')


if __name__=="__main__":
    batch_size_train=1
    batch_size_infer=1
    epochs=100
    patience=5
    num_workers = 6

    train_dataset = NpyDataset('../data/Xtrain_base.npy', '../data/Ytrain_base.npy')
    val_dataset = NpyDataset('../data/Xval_base.npy', '../data/Yval_base.npy')
    vocab = pd.read_csv('../data/vocab.csv', header=None, index_col=0)
    vocab_size = vocab.shape[0]
    cat_label_mapping = pd.read_csv('../data/cat_label_mapping.csv', header=None)
    num_classes = cat_label_mapping.shape[0]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = Transformer(num_layers=6,
                    vocab_size=vocab_size,
                    d_model=256,
                    d_q_k_v=64,
                    num_heads=6,
                    num_classes=num_classes,
                    hidden_dim=16,
                    dropout=0.1
                    )

    net.to(device=device)
    train_model(net, train_dataset, val_dataset, epochs, batch_size_train, batch_size_infer,patience,num_workers)


