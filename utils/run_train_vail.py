import os
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
from utils.utils import calculate_metrics
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import torch.nn.functional as F


def uncertain_loss(prob_fg, prob_bg, prob_uc):
    loss = (prob_fg * prob_bg).sum() + (prob_fg * prob_uc).sum() + (prob_bg * prob_uc).sum() + (prob_uc).sum()
    num_pixels = prob_fg.size(0) * prob_fg.size(2) * prob_fg.size(3)
    normalized_loss = loss / num_pixels
    return normalized_loss

def train(model, loader, optimizer, loss_fn, device):
    model.train()

    metrics = {
        'loss': 0.0,
        'miou': 0.0,
        'dsc': 0.0,
        'acc': 0.0,
        'sen': 0.0,
        'spe': 0.0,
        'pre': 0.0,
        'rec': 0.0,
        'fb': 0.0,
        'em': 0.0
    }

    for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        
        x = x.to(device, dtype=torch.float32)
        y1 = y1.to(device, dtype=torch.float32)
        y2 = y2.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        
        mask_pred, fg_pred, bg_pred, uc_pred = model(x)
        
        loss_mask =  loss_fn(mask_pred, y1)
        loss_fg = loss_fn(fg_pred[0], y1) + loss_fn(fg_pred[1], y1) +  loss_fn(fg_pred[2], y1) 
        loss_bg = loss_fn(bg_pred[0], y2) + loss_fn(bg_pred[1], y2) + loss_fn(bg_pred[2], y2)
        
        loss_comp = []
        for i in range(3):
            preds = torch.stack([fg_pred[i], bg_pred[i], uc_pred[i]], dim=1)
            probs = F.softmax(preds, dim=1)
            prob_fg, prob_bg, prob_uc = probs[:, 0], probs[:, 1], probs[:, 2]
            loss_comp.append(uncertain_loss(prob_fg, prob_bg, prob_uc).to(device))
        loss_comp = loss_comp[0]+ loss_comp[1]+ loss_comp[2]
        
        loss =  loss_mask + loss_fg + loss_bg + loss_comp
        loss.backward()
        optimizer.step()

        metrics['loss'] += loss.item()

def evaluate(model, loader, loss_fn, device):
    model.eval()

    metrics = {
        'loss': 0.0,
        'miou': 0.0,
        'dsc': 0.0,
        'acc': 0.0,
        'sen': 0.0,
        'spe': 0.0,
        'pre': 0.0,
        'rec': 0.0,
        'fb': 0.0,
        'em': 0.0
    }

    with torch.no_grad():
        for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Evaluation", total=len(loader))):
            x = x.to(device, dtype=torch.float32)
            y1 = y1.to(device, dtype=torch.float32)
            y2 = y2.to(device, dtype=torch.float32)

            mask_pred, _, _, _ = model(x)
            loss = loss_fn(mask_pred, y1)

            metrics['loss'] += loss.item()

            batch_metrics = {
                'miou': [], 'dsc': [], 'acc': [], 'sen': [],
                'spe': [], 'pre': [], 'rec': [], 'fb': [], 'em': []
            }

            for yt, yp in zip(y1, mask_pred):
                scores = calculate_metrics(yt, yp)
                for idx, key in enumerate(batch_metrics.keys()):
                    batch_metrics[key].append(scores[idx])

            for key in batch_metrics:
                metrics[key] += np.mean(batch_metrics[key])

    for key in metrics:
        metrics[key] /= len(loader)

    return metrics['loss'], [
        metrics['miou'], metrics['dsc'], metrics['acc'], 
        metrics['sen'], metrics['spe'], metrics['pre'],
        metrics['rec'], metrics['fb'], metrics['em']
    ]

