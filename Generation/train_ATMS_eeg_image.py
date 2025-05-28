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
    def __init__(self, data_path, train=True):
        self.data_path = data_path
        self.train = train
        self.data = np.load(data_path, allow_pickle=True).item()
        self.X = self.data['X_train'] if train else self.data['X_test']
        self.y = self.data['y_train'] if train else self.data['y_test']
        self.y_classes = np.vectorize(lambda x: x.split('_')[0])(self.y)

        if train:
            n_trials, n_repeats, n_channels, n_timepoints = self.X.shape
            self.X = einops.rearrange(self.X, 'n r c t -> (n r) c t')
            self.y = einops.repeat(self.y, 'n -> (n r)', r=n_repeats)
            self.y_classes = einops.repeat(self.y_classes, 'n -> (n r)', r=n_repeats)
        else:
            self.X = np.mean(self.X, axis=1)

        print(f"EEG data shape: {self.X.shape}")

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
    parser = argparse.ArgumentParser(description='ATMS Training for EEG-Image Dataset')
    parser.add_argument('--experiment_id', type=str, default='gtec_250527_data_250527', help='Experiment ID')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--save_every', type=int, default=5, help='Save model every N epochs')
    args = parser.parse_args()

    # Setup paths
    experiment_folder = f"./experiment_{args.experiment_id}"
    data_path = os.path.join(experiment_folder, 'whitened_eeg_data.npy')
    stimuli_folder = os.path.join(experiment_folder, 'image_pool')
    
    # Create output directories
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    model_dir = os.path.join(experiment_folder, 'models', 'ATMS', current_time)
    results_dir = os.path.join(experiment_folder, 'training_results', current_time)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    print('Loading datasets...')
    train_dataset = EEGImageDataset(data_path, train=True)
    test_dataset = EEGImageDataset(data_path, train=False)

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

    optimizer = AdamW(eeg_model.parameters(), lr=args.lr)

    # Training loop
    print('Starting training...')
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
    results_file = os.path.join(results_dir, "training_results.csv")
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
    info_text = "Best Model Info:\n"
    for key, value in best_epoch_info.items():
        if isinstance(value, float):
            info_text += f"{key}: {value:.4f}\n"
        else:
            info_text += f"{key}: {value}\n"
    axes[1, 1].text(0.1, 0.9, info_text, transform=axes[1, 1].transAxes, 
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