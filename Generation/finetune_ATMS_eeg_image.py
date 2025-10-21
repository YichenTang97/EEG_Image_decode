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
import open_clip
from PIL import Image

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

class EEGImageDataset(Dataset):
    def __init__(self, data_path, train=True, pretrained_channels=None, data_channels=None):
        self.data_path = data_path
        self.train = train
        self.pretrained_channels = pretrained_channels
        self.data_channels = data_channels
        
        self.data = np.load(data_path, allow_pickle=True).item()
        self.X = self.data['X_train'] if train else self.data['X_test']
        self.y = self.data['y_train'] if train else self.data['y_test']
        self.y_classes = np.vectorize(lambda x: x.split('_')[0])(self.y)

        # Handle channel selection and reordering
        if pretrained_channels is not None and data_channels is not None:
            self.channel_mapping = self._create_channel_mapping()
            print(f"Channel mapping created: {len(self.channel_mapping)} channels selected")
            print(f"Selected channels: {pretrained_channels}")
        else:
            self.channel_mapping = None
            print("No channel mapping - using all available channels")

        if train:
            n_trials, n_repeats, n_channels, n_timepoints = self.X.shape
            self.X = einops.rearrange(self.X, 'n r c t -> (n r) c t')
            self.y = einops.repeat(self.y, 'n -> (n r)', r=n_repeats)
            self.y_classes = einops.repeat(self.y_classes, 'n -> (n r)', r=n_repeats)
        else:
            self.X = np.mean(self.X, axis=1)

        # Apply channel selection if mapping exists
        if self.channel_mapping is not None:
            self.X = self.X[:, self.channel_mapping, :]
            print(f"Applied channel selection. New EEG data shape: {self.X.shape}")

        print(f"Final EEG data shape: {self.X.shape}")

    def _create_channel_mapping(self):
        """
        Create mapping from data channels to pretrained channels.
        Returns indices for selecting and reordering channels.
        """
        if self.pretrained_channels is None or self.data_channels is None:
            return None
            
        # Create mapping from channel names to indices
        data_channel_to_idx = {ch: idx for idx, ch in enumerate(self.data_channels)}
        
        # Find indices for pretrained channels in the data
        channel_mapping = []
        missing_channels = []
        
        for pretrained_ch in self.pretrained_channels:
            if pretrained_ch in data_channel_to_idx:
                channel_mapping.append(data_channel_to_idx[pretrained_ch])
            else:
                missing_channels.append(pretrained_ch)
        
        if missing_channels:
            print(f"WARNING: {len(missing_channels)} pretrained channels not found in data: {missing_channels}")
            print(f"Available data channels: {self.data_channels}")
        
        if len(channel_mapping) == 0:
            raise ValueError("No matching channels found between pretrained model and data!")
        
        print(f"Successfully mapped {len(channel_mapping)}/{len(self.pretrained_channels)} pretrained channels")
        return channel_mapping

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        eeg_data = self.X[idx]
        label = self.y[idx]
        label_class = self.y_classes[idx]
        return torch.tensor(eeg_data, dtype=torch.float32), label, label_class

def precompute_image_embeddings(stimuli_folder, labels, device):
    """
    Precompute image embeddings for all unique labels to save time during training.
    """
    print('Precomputing image embeddings...')
    model, preprocess, _ = open_clip.create_model_and_transforms('ViT-H-14',
            pretrained="./variables/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
            precision='fp32', device=device)

    # Map label to image path
    label_to_image = {img.split('.')[0]: os.path.join(stimuli_folder, img) 
                     for img in os.listdir(stimuli_folder) 
                     if img.lower().endswith(('.png', '.jpg', '.jpeg'))}

    # Compute unique image embeddings
    unique_labels = sorted(list(set(labels)))  # Ensure deterministic order
    unique_image_features = {}
    missing_labels = []

    for label in unique_labels:
        if label not in label_to_image:
            missing_labels.append(label)
            print(f"WARNING: No image found for label '{label}'")
            continue

        image_path = label_to_image[label]
        try:
            image_input = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                image_feature = model.encode_image(image_input)
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
            unique_image_features[label] = image_feature.squeeze(0).cpu()  # Store on CPU to save GPU memory
        except Exception as e:
            print(f"ERROR loading image for label '{label}' at path '{image_path}': {e}")
            missing_labels.append(label)

    if missing_labels:
        print(f"WARNING: {len(missing_labels)} labels have missing or corrupted images: {missing_labels}")

    # Create label to embedding mapping
    valid_labels = [label for label in unique_labels if label in unique_image_features]
    print(f"Successfully precomputed embeddings for {len(valid_labels)}/{len(unique_labels)} labels")
    
    return unique_image_features, valid_labels

def get_batch_image_embeddings(labels, precomputed_embeddings, device):
    """
    Get image embeddings for a batch of labels from precomputed embeddings.
    """
    batch_embeddings = []
    valid_indices = []
    
    for i, label in enumerate(labels):
        if label in precomputed_embeddings:
            batch_embeddings.append(precomputed_embeddings[label])
            valid_indices.append(i)
    
    if batch_embeddings:
        return torch.stack(batch_embeddings).to(device), valid_indices
    else:
        return torch.empty(0, 1024).to(device), []

def load_pretrained_model(model, pretrained_path, device):
    """
    Load pretrained model weights with proper error handling.
    """
    print(f"Loading pretrained model from: {pretrained_path}")
    
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained model file not found: {pretrained_path}")
    
    try:
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load the state dict
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded pretrained weights!")
        
        # Print which layers were loaded
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        loaded_keys = model_keys.intersection(checkpoint_keys)
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        print(f"Loaded {len(loaded_keys)} layers from pretrained model")
        if missing_keys:
            print(f"Missing keys (will use random initialization): {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {len(unexpected_keys)}")
            
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        print("Continuing with random initialization...")

def train_model(eeg_model, dataloader, optimizer, device, precomputed_embeddings):
    eeg_model.train()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.90
    mse_loss_fn = nn.MSELoss()

    for batch_idx, (eeg_data, labels, classes) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        
        # Get precomputed image embeddings for this batch
        img_features, valid_indices = get_batch_image_embeddings(labels, precomputed_embeddings, device)
        
        # Filter EEG data to match valid labels
        if len(valid_indices) != len(labels):
            eeg_data = eeg_data[valid_indices]
            labels = [labels[i] for i in valid_indices]
        
        if len(eeg_data) == 0:
            continue
            
        img_features = img_features.float()

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

        # Compute accuracy
        logits_img = logit_scale * eeg_features @ img_features.T
        predicted = torch.argmax(logits_img, dim=1)
        correct += (predicted == torch.arange(len(eeg_features)).to(device)).sum().item()
        total += len(eeg_features)

    average_loss = total_loss / (batch_idx + 1) if batch_idx >= 0 else 0
    accuracy = correct / total if total > 0 else 0
    return average_loss, accuracy

def evaluate_model(eeg_model, dataloader, device, precomputed_embeddings, valid_labels, k_values=[2, 4, 10, 50, 100]):
    eeg_model.eval()
    results = {}
    
    # Create embeddings tensor for all valid labels
    img_features_all = torch.stack([precomputed_embeddings[label] for label in valid_labels]).to(device).float()
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(valid_labels)}
    
    for k in k_values:
        if k > len(valid_labels):
            k = len(valid_labels)
            
        correct = 0
        total = 0
        top5_correct = 0
        
        with torch.no_grad():
            for eeg_data, labels, classes in dataloader:
                eeg_data = eeg_data.to(device)
                batch_size = eeg_data.size(0)
                subject_ids = torch.full((batch_size,), 1, dtype=torch.long).to(device)
                eeg_features = eeg_model(eeg_data, subject_ids)
                
                logit_scale = eeg_model.logit_scale
                
                for idx, label in enumerate(labels):
                    if label not in label_to_idx:
                        continue
                        
                    true_label_idx = label_to_idx[label]
                    
                    # Select k classes for evaluation
                    if k >= len(valid_labels):
                        selected_indices = list(range(len(valid_labels)))
                    else:
                        possible_indices = list(range(len(valid_labels)))
                        possible_indices.remove(true_label_idx)
                        selected_indices = random.sample(possible_indices, k-1) + [true_label_idx]
                    
                    selected_img_features = img_features_all[selected_indices]
                    logits = logit_scale * eeg_features[idx] @ selected_img_features.T
                    
                    predicted_idx = torch.argmax(logits).item()
                    if selected_indices[predicted_idx] == true_label_idx:
                        correct += 1
                    
                    # Top-5 accuracy
                    if k >= 5:
                        _, top5_indices = torch.topk(logits, min(5, len(selected_indices)), largest=True)
                        if true_label_idx in [selected_indices[i] for i in top5_indices.tolist()]:
                            top5_correct += 1
                    
                    total += 1
        
        accuracy = correct / total if total > 0 else 0
        top5_accuracy = top5_correct / total if total > 0 and k >= 5 else 0
        
        results[f'k={k}_accuracy'] = accuracy
        if k >= 5:
            results[f'k={k}_top5_accuracy'] = top5_accuracy
    
    return results

def main():
    parser = argparse.ArgumentParser(description='ATMS Finetuning for EEG-Image Dataset')
    parser.add_argument('--experiment_id', type=str, default='gtec_250627_251006_finetune', help='Experiment ID')
    parser.add_argument('--data_file', type=str, default='whitened_eeg_data.npy', help='Name of the EEG data file (default: whitened_eeg_data.npy)')
    parser.add_argument('--pretrained_model', type=str, required=True, help='Path to pretrained model checkpoint')
    parser.add_argument('--pretrained_channels', type=str, help='Path to file containing channels used during pretraining (one channel per line)')
    parser.add_argument('--data_channels', type=str, help='Path to file containing channels in your EEG data (one channel per line)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for finetuning (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for finetuning (default: 20)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--save_every', type=int, default=5, help='Save model every N epochs')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze the encoder layers during finetuning')
    args = parser.parse_args()

    # Setup paths
    experiment_folder = f"./experiments/experiment_{args.experiment_id}"
    data_path = os.path.join(experiment_folder, args.data_file)
    stimuli_folder = os.path.join(experiment_folder, 'image_pool')
    
    # Create output directories
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    model_dir = os.path.join(experiment_folder, 'models', 'ATMS_finetuned', current_time)
    results_dir = os.path.join(experiment_folder, 'finetuning_results', current_time)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load channel configurations
    pretrained_channels = None
    data_channels = None
    
    if args.pretrained_channels:
        print(f"Loading pretrained channels from: {args.pretrained_channels}")
        with open(args.pretrained_channels, 'r') as f:
            pretrained_channels = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Pretrained channels ({len(pretrained_channels)}): {pretrained_channels}")
    
    if args.data_channels:
        print(f"Loading data channels from: {args.data_channels}")
        with open(args.data_channels, 'r') as f:
            data_channels = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Data channels ({len(data_channels)}): {data_channels}")

    # Load datasets
    print('Loading datasets...')
    train_dataset = EEGImageDataset(data_path, train=True, pretrained_channels=pretrained_channels, data_channels=data_channels)
    test_dataset = EEGImageDataset(data_path, train=False, pretrained_channels=pretrained_channels, data_channels=data_channels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Precompute image embeddings
    print('Precomputing image embeddings...')
    all_labels = list(train_dataset.y) + list(test_dataset.y)
    precomputed_embeddings, valid_labels = precompute_image_embeddings(stimuli_folder, all_labels, device)

    # Initialize model
    print('Initializing ATMS model...')
    eeg_model = ATMS(num_channels=train_dataset.X.shape[1], sequence_length=250)
    eeg_model.to(device)

    # Load pretrained weights
    load_pretrained_model(eeg_model, args.pretrained_model, device)

    # Optionally freeze encoder layers
    if args.freeze_encoder:
        print("Freezing encoder layers...")
        for param in eeg_model.encoder.parameters():
            param.requires_grad = False

    # Setup optimizer with different learning rates for different parts
    if args.freeze_encoder:
        # Only optimize non-frozen parameters
        optimizer = AdamW(filter(lambda p: p.requires_grad, eeg_model.parameters()), lr=args.lr)
    else:
        # Use different learning rates for different parts
        encoder_params = list(eeg_model.encoder.parameters())
        other_params = list(eeg_model.enc_eeg.parameters()) + list(eeg_model.proj_eeg.parameters()) + [eeg_model.logit_scale]
        
        optimizer = AdamW([
            {'params': encoder_params, 'lr': args.lr * 0.1},  # Lower LR for pretrained encoder
            {'params': other_params, 'lr': args.lr}           # Higher LR for other layers
        ])

    # Training loop
    print('Starting finetuning...')
    train_losses, train_accuracies = [], []
    test_results_history = []
    best_accuracy = 0.0
    best_epoch_info = {}

    for epoch in tqdm(range(args.epochs)):
        # Train
        train_loss, train_accuracy = train_model(eeg_model, train_loader, optimizer, device, precomputed_embeddings)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate
        test_results = evaluate_model(eeg_model, test_loader, device, precomputed_embeddings, valid_labels)
        test_results_history.append(test_results)

        # Save model periodically
        if (epoch + 1) % args.save_every == 0:
            model_path = os.path.join(model_dir, f"epoch_{epoch+1}.pth")
            torch.save(eeg_model.state_dict(), model_path)
            print(f"Model saved: {model_path}")

        # Track best model
        current_accuracy = test_results.get('k=100_accuracy', 0)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_epoch_info = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                **test_results
            }
            # Save best model
            best_model_path = os.path.join(model_dir, "best_model.pth")
            torch.save(eeg_model.state_dict(), best_model_path)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        for k, acc in test_results.items():
            print(f"Test {k}: {acc:.4f}")
        print("-" * 50)

    # Save final model
    final_model_path = os.path.join(model_dir, "final_model.pth")
    torch.save(eeg_model.state_dict(), final_model_path)

    # Save training results
    results_file = os.path.join(results_dir, "finetuning_results.csv")
    with open(results_file, 'w', newline='') as file:
        fieldnames = ['epoch', 'train_loss', 'train_accuracy'] + list(test_results_history[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for epoch, test_results in enumerate(test_results_history):
            row = {
                'epoch': epoch + 1,
                'train_loss': train_losses[epoch],
                'train_accuracy': train_accuracies[epoch],
                **test_results
            }
            writer.writerow(row)

    # Save best model info
    best_info_file = os.path.join(results_dir, "best_model_info.txt")
    with open(best_info_file, 'w') as f:
        f.write("Best Finetuned Model Information\n")
        f.write("=" * 40 + "\n")
        f.write(f"Experiment ID: {args.experiment_id}\n")
        f.write(f"Data file: {args.data_file}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Pretrained model: {args.pretrained_model}\n")
        f.write(f"Finetuning epochs: {args.epochs}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Frozen encoder: {args.freeze_encoder}\n")
        if pretrained_channels:
            f.write(f"Pretrained channels: {len(pretrained_channels)} channels\n")
            f.write(f"Pretrained channels file: {args.pretrained_channels}\n")
        if data_channels:
            f.write(f"Data channels: {len(data_channels)} channels\n")
            f.write(f"Data channels file: {args.data_channels}\n")
        f.write(f"Final model channels: {train_dataset.X.shape[1]}\n")
        f.write("\nBest Results:\n")
        for key, value in best_epoch_info.items():
            f.write(f"{key}: {value}\n")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training curves
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].set_title('Finetuning Loss')
    axes[0, 0].legend()
    
    axes[0, 1].plot(train_accuracies, label='Train Accuracy')
    axes[0, 1].set_title('Finetuning Accuracy')
    axes[0, 1].legend()
    
    # Test accuracies
    k_values = [2, 4, 10, 50, 100]
    for k in k_values:
        key = f'k={k}_accuracy'
        if key in test_results_history[0]:
            accuracies = [results[key] for results in test_results_history]
            axes[1, 0].plot(accuracies, label=f'k={k}')
    axes[1, 0].set_title('Test Accuracies')
    axes[1, 0].legend()
    
    # Best model info
    axes[1, 1].axis('off')
    info_text = "Best Finetuned Model Info:\n"
    info_text += f"Data file: {args.data_file}\n"
    info_text += f"Pretrained: {os.path.basename(args.pretrained_model)}\n"
    info_text += f"Epochs: {args.epochs}\n"
    info_text += f"LR: {args.lr}\n"
    info_text += f"Frozen encoder: {args.freeze_encoder}\n\n"
    for key, value in best_epoch_info.items():
        if isinstance(value, float):
            info_text += f"{key}: {value:.4f}\n"
        else:
            info_text += f"{key}: {value}\n"
    axes[1, 1].text(0.1, 0.9, info_text, transform=axes[1, 1].transAxes, 
                    verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "finetuning_plots.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"\nFinetuning completed!")
    print(f"Models saved in: {model_dir}")
    print(f"Results saved in: {results_dir}")
    print(f"Best accuracy: {best_accuracy:.4f} at epoch {best_epoch_info['epoch']}")

if __name__ == "__main__":
    main()
