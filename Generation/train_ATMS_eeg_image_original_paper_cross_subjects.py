import os
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from einops.layers.torch import Rearrange, Reduce
import einops
from sklearn.metrics import confusion_matrix
import random
import csv
from torch import Tensor
import itertools
import math
import re
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
from loss import ClipLoss
import argparse
from torch.optim import AdamW
import datetime
import matplotlib.pyplot as plt
from eegdatasets_leaveone import EEGDataset

class Config:
    def __init__(self):
        self.task_name = 'classification'
        self.seq_len = 250
        self.pred_len = 250
        self.output_attention = False
        self.d_model = 250
        self.embed = 'timeF'
        self.freq = 'h'
        self.dropout = 0.25
        self.factor = 1
        self.n_heads = 4
        self.e_layers = 1
        self.d_ff = 256
        self.activation = 'gelu'
        self.enc_in = 63

class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=10, num_channels=63):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.num_channels = num_channels
        # Embedding
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout, joint_train=False, num_subjects=num_subjects)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :self.num_channels, :]
        return enc_out

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, num_channels=63):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (num_channels, 1), stride=(1, 1)),  # Adjust kernel size dynamically
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = self.tsconv(x)
        x = self.projection(x)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, num_channels=63, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size, num_channels),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class ATMS(nn.Module):
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=2, num_features=64, num_latents=1024, num_blocks=1):
        super(ATMS, self).__init__()
        default_config = Config()
        self.encoder = iTransformer(default_config, num_channels=num_channels)
        # Fix: Remove unused subject_wise_linear - it was never used properly anyway
        self.enc_eeg = Enc_eeg(num_channels=num_channels)
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        return out

# EEGImageDataset removed - using EEGDataset from eegdatasets_leaveone.py instead

# Precomputed image embeddings functions removed - EEGDataset handles this internally

def train_model(eeg_model, dataloader, optimizer, device, text_features_all, img_features_all):
    eeg_model.train()
    text_features_all = text_features_all.to(device).float()
    img_features_all = (img_features_all[::10]).to(device).float()  # Subsample like original
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.90
    mse_loss_fn = nn.MSELoss()

    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()

        batch_size = eeg_data.size(0)
        subject_ids = torch.full((batch_size,), 1, dtype=torch.long).to(device)
        eeg_features = eeg_model(eeg_data, subject_ids).float()

        logit_scale = eeg_model.logit_scale

        img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
        regress_loss = mse_loss_fn(eeg_features, img_features)
        loss = (alpha * regress_loss * 10 + (1 - alpha) * img_loss * 10)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        # Compute accuracy using subsampled features (like original)
        logits_img = logit_scale * eeg_features @ img_features_all.T
        predicted = torch.argmax(logits_img, dim=1)
        correct += (predicted == labels).sum().item()
        total += len(eeg_features)

    average_loss = total_loss / (batch_idx + 1) if batch_idx >= 0 else 0
    accuracy = correct / total if total > 0 else 0
    return average_loss, accuracy

# Evaluation function removed - only storing training losses

def main():
    parser = argparse.ArgumentParser(description='ATMS Training for EEG-Image Dataset')
    parser.add_argument('--data_path', type=str, default="./../Preprocessed_data_250Hz", help='Path to the EEG dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/pretrained/all_subject_250929', help='Directory to save output results')    
    parser.add_argument('--channels_conf', type=str, default='./variables/channels_biosemi_59.txt', help='Configuration file for EEG channels to use')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--save_every', type=int, default=5, help='Save model every N epochs')
    parser.add_argument('--subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'], help='List of subject IDs (default: sub-01 to sub-10)')    
    args = parser.parse_args()

    # Setup paths
    data_path = args.data_path
    
    # Create output directories
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    model_dir = os.path.join(args.output_dir, 'ATMS', 'all_subjects', current_time)
    results_dir = os.path.join(args.output_dir, 'training_results', current_time)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    print('Loading datasets...')
    channels = None
    if args.channels_conf != 'none':
        with open(args.channels_conf, 'r') as f:
            channels = [line.strip() for line in f.readlines()]
        print(f'Using channels configuration from {args.channels_conf}')
        print(f'Channels used: {channels}')
    num_channels = len(channels) if args.channels_conf != 'none' else 63  # Default to 63 channels if none specified

    train_dataset = EEGDataset(data_path, subjects=args.subjects, train=True, channels=channels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Initialize model
    print('Initializing ATMS model...')
    eeg_model = ATMS(num_channels=num_channels, sequence_length=250)
    eeg_model.to(device)

    optimizer = AdamW(eeg_model.parameters(), lr=args.lr)

    # Training loop
    print('Starting training...')
    train_losses, train_accuracies = [], []
    best_accuracy = 0.0
    best_epoch_info = {}

    for epoch in tqdm(range(args.epochs)):
        # Train
        train_loss, train_accuracy = train_model(eeg_model, train_loader, optimizer, device, train_dataset.text_features, train_dataset.img_features)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Save model periodically
        if (epoch + 1) % args.save_every == 0:
            model_path = os.path.join(model_dir, f"epoch_{epoch+1}.pth")
            torch.save(eeg_model.state_dict(), model_path)
            print(f"Model saved: {model_path}")

        # Track best model based on training accuracy
        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            best_epoch_info = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
            }
            # Save best model
            best_model_path = os.path.join(model_dir, "best_model.pth")
            torch.save(eeg_model.state_dict(), best_model_path)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print("-" * 50)

    # Save final model
    final_model_path = os.path.join(model_dir, "final_model.pth")
    torch.save(eeg_model.state_dict(), final_model_path)

    # Save training results
    results_file = os.path.join(results_dir, "training_results.csv")
    with open(results_file, 'w', newline='') as file:
        fieldnames = ['epoch', 'train_loss', 'train_accuracy']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for epoch in range(len(train_losses)):
            row = {
                'epoch': epoch + 1,
                'train_loss': train_losses[epoch],
                'train_accuracy': train_accuracies[epoch],
            }
            writer.writerow(row)

    # Save best model info
    best_info_file = os.path.join(results_dir, "best_model_info.txt")
    with open(best_info_file, 'w') as f:
        f.write("Best Model Information\n")
        f.write("=" * 30 + "\n")
        for key, value in best_epoch_info.items():
            f.write(f"{key}: {value}\n")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training curves
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    
    axes[0, 1].plot(train_accuracies, label='Train Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].legend()
    
    # Best model info
    axes[1, 0].axis('off')
    info_text = "Best Model Info:\n"
    for key, value in best_epoch_info.items():
        if isinstance(value, float):
            info_text += f"{key}: {value:.4f}\n"
        else:
            info_text += f"{key}: {value}\n"
    axes[1, 0].text(0.1, 0.9, info_text, transform=axes[1, 0].transAxes, 
                    verticalalignment='top', fontsize=10)
    
    # Training summary
    axes[1, 1].axis('off')
    summary_text = f"Training Summary:\n"
    summary_text += f"Total Epochs: {len(train_losses)}\n"
    summary_text += f"Final Loss: {train_losses[-1]:.4f}\n"
    summary_text += f"Final Accuracy: {train_accuracies[-1]:.4f}\n"
    summary_text += f"Best Accuracy: {best_accuracy:.4f}\n"
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                    verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "training_plots.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"\nTraining completed!")
    print(f"Models saved in: {model_dir}")
    print(f"Results saved in: {results_dir}")
    print(f"Best accuracy: {best_accuracy:.4f} at epoch {best_epoch_info['epoch']}")

if __name__ == "__main__":
    main() 