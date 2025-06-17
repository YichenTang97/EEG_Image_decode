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
    def __init__(self, d_model=250, n_heads=4, e_layers=1, d_ff=256, dropout=0.25, factor=1, seq_len=250):
        self.task_name = 'classification'
        self.seq_len = seq_len
        self.pred_len = seq_len
        self.output_attention = False
        self.d_model = d_model
        self.embed = 'timeF'
        self.freq = 'h'
        self.dropout = dropout
        self.factor = factor
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.activation = 'gelu'
        self.enc_in = None  # Will be set based on number of channels

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
    def __init__(self, emb_size=40, num_channels=63, seq_len=250):
        super().__init__()
        # Calculate the output size after convolutions and pooling
        # First conv: (1, 25) -> seq_len - 24
        # AvgPool: (1, 51) -> (seq_len - 24 - 50) / 5
        # Second conv: (num_channels, 1) -> 1
        # Final projection: (1, 1) -> emb_size
        
        # Calculate the number of timepoints after processing
        temp_len = seq_len - 24  # After first conv
        temp_len = math.ceil((temp_len - 50) / 5)  # After avgpool - rounded up
        temp_len = temp_len  # After second conv (no change in time dimension)
        
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (num_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        # Store the expected output size for flattening
        self.output_size = temp_len * emb_size

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
    def __init__(self, emb_size=40, num_channels=63, seq_len=250, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size, num_channels, seq_len),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

# class AttentionCombiner(nn.Module):
#     def __init__(self, embedding_dim):
#         super(AttentionCombiner, self).__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1)
    
#     def forward(self, emb1, emb2):
#         combined = torch.stack((emb1, emb2), dim=0)
#         attention_output, _ = self.attention(combined, combined, combined)
#         return torch.mean(attention_output, dim=0)

class ATMS_Multimodal(nn.Module):
    def __init__(self, eeg_channels=15, fnirs_channels=16, eeg_seq_len=250, fnirs_seq_len=200, num_subjects=2, num_features=64, num_latents=1024, num_blocks=1, no_fnirs=False, no_eeg=False):
        super(ATMS_Multimodal, self).__init__()
        
        self.no_fnirs = no_fnirs
        self.no_eeg = no_eeg

        d_model_eeg = eeg_seq_len
        d_model_fnirs = fnirs_seq_len
        
        # Calculate embedding dimensions for each modality
        eeg_temp_len = d_model_eeg - 24
        eeg_temp_len = math.ceil((eeg_temp_len - 50) / 5)
        eeg_embedding_dim = eeg_temp_len * 40  # 40 is the emb_size
        
        # EEG stream (only if not no_eeg)
        if not self.no_eeg:
            # EEG config
            eeg_config = Config(
                d_model=d_model_eeg,  # Model dimension equals sequence length
                n_heads=4,    # Number of attention heads
                e_layers=1,   # Number of encoder layers
                d_ff=256,     # Feed-forward dimension
                dropout=0.25, # Dropout rate
                factor=1,     # Attention factor
                seq_len=eeg_seq_len
            )
            eeg_config.enc_in = eeg_channels
            
            self.eeg_encoder = iTransformer(eeg_config, num_channels=eeg_channels)
            self.enc_eeg = Enc_eeg(num_channels=eeg_channels, seq_len=d_model_eeg)
            self.proj_eeg = Proj_eeg(embedding_dim=eeg_embedding_dim)
        
        # fNIRS stream (only if not no_fnirs)
        if not self.no_fnirs:
            fnirs_temp_len = d_model_fnirs - 24
            fnirs_temp_len = math.ceil((fnirs_temp_len - 50) / 5)
            fnirs_embedding_dim = fnirs_temp_len * 40  # 40 is the emb_size
            
            # fNIRS config
            fnirs_config = Config(
                d_model=d_model_fnirs,  # Model dimension equals sequence length
                n_heads=2,    # Number of attention heads
                e_layers=1,   # Number of encoder layers
                d_ff=256,     # Feed-forward dimension
                dropout=0.5, # Dropout rate
                factor=1,     # Attention factor
                seq_len=fnirs_seq_len
            )
            fnirs_config.enc_in = fnirs_channels
            
            self.fnirs_encoder = iTransformer(fnirs_config, num_channels=fnirs_channels)
            self.enc_fnirs = Enc_eeg(num_channels=fnirs_channels, seq_len=d_model_fnirs)
            self.proj_fnirs = Proj_eeg(embedding_dim=fnirs_embedding_dim)
        
        # Fusion layer (only needed if both modalities are used)
        if not self.no_eeg and not self.no_fnirs:
            # Simple averaging fusion for multimodal
            # self.attention_combiner = AttentionCombiner(embedding_dim=1024)
            
            # Layer normalization after fusion
            self.fusion_norm = nn.LayerNorm(1024)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, eeg_data, fnirs_data, subject_ids):
        if not self.no_eeg and not self.no_fnirs:
            # Process both EEG and fNIRS streams
            eeg_encoded = self.eeg_encoder(eeg_data, None, subject_ids)
            eeg_embedding = self.enc_eeg(eeg_encoded)
            eeg_proj = self.proj_eeg(eeg_embedding)
            
            fnirs_encoded = self.fnirs_encoder(fnirs_data, None, subject_ids)
            fnirs_embedding = self.enc_fnirs(fnirs_encoded)
            fnirs_proj = self.proj_fnirs(fnirs_embedding)
            
            # Combine both streams using simple averaging
            # out = self.attention_combiner(eeg_proj, fnirs_proj)  # For attention method
            out = (eeg_proj + fnirs_proj) / 2  # For simple averaging
            
            # Apply layer normalization after fusion
            out = self.fusion_norm(out)
            
        elif self.no_eeg and not self.no_fnirs:
            # Use only fNIRS data
            fnirs_encoded = self.fnirs_encoder(fnirs_data, None, subject_ids)
            fnirs_embedding = self.enc_fnirs(fnirs_encoded)
            out = self.proj_fnirs(fnirs_embedding)
            
        elif not self.no_eeg and self.no_fnirs:
            # Use only EEG data
            eeg_encoded = self.eeg_encoder(eeg_data, None, subject_ids)
            eeg_embedding = self.enc_eeg(eeg_encoded)
            out = self.proj_eeg(eeg_embedding)
            
        else:
            raise ValueError("Both no_eeg and no_fnirs cannot be True simultaneously")
            
        return out

class EEGfNIRSImageDataset(Dataset):
    def __init__(self, eeg_data_path, fnirs_data_path, train=True):
        self.train = train
        
        # Load EEG data if provided
        if eeg_data_path is not None:
            self.eeg_data = np.load(eeg_data_path, allow_pickle=True).item()
            self.eeg_X = self.eeg_data['X_train'] if train else self.eeg_data['X_test']
            self.y = self.eeg_data['y_train'] if train else self.eeg_data['y_test']
            self.y_classes = np.vectorize(lambda x: x.split('_')[0])(self.y)
        else:
            # Create dummy EEG data with same shape as fNIRS
            self.eeg_X = None
            self.y = None
            self.y_classes = None
        
        # Load fNIRS data if provided
        if fnirs_data_path is not None:
            self.fnirs_data = np.load(fnirs_data_path, allow_pickle=True).item()
            self.fnirs_X = self.fnirs_data['X_train'] if train else self.fnirs_data['X_test']
            # Use fNIRS labels if EEG is not available
            if self.y is None:
                self.y = self.fnirs_data['y_train'] if train else self.fnirs_data['y_test']
                self.y_classes = np.vectorize(lambda x: x.split('_')[0])(self.y)
        else:
            # Create dummy fNIRS data with same shape as EEG
            if self.eeg_X is not None:
                self.fnirs_X = np.zeros_like(self.eeg_X)
            else:
                raise ValueError("At least one of eeg_data_path or fnirs_data_path must be provided")
        
        if train:
            # Reshape EEG data if available
            if self.eeg_X is not None:
                n_trials, n_repeats, n_channels, n_timepoints = self.eeg_X.shape
                self.eeg_X = einops.rearrange(self.eeg_X, 'n r c t -> (n r) c t')
            
            # Reshape fNIRS data
            n_trials, n_repeats, n_channels, n_timepoints = self.fnirs_X.shape
            self.fnirs_X = einops.rearrange(self.fnirs_X, 'n r c t -> (n r) c t')
            
            # Repeat labels
            self.y = einops.repeat(self.y, 'n -> (n r)', r=n_repeats)
            self.y_classes = einops.repeat(self.y_classes, 'n -> (n r)', r=n_repeats)
        else:
            if self.eeg_X is not None:
                self.eeg_X = np.mean(self.eeg_X, axis=1)
            self.fnirs_X = np.mean(self.fnirs_X, axis=1)

        if self.eeg_X is not None:
            print(f"EEG data shape: {self.eeg_X.shape}")
        print(f"fNIRS data shape: {self.fnirs_X.shape}")

    def __len__(self):
        return len(self.fnirs_X)  # Use fNIRS length as reference

    def __getitem__(self, idx):
        if self.eeg_X is not None:
            eeg_data = self.eeg_X[idx]
        else:
            # Create dummy EEG data with same shape as fNIRS for compatibility
            eeg_data = np.zeros_like(self.fnirs_X[idx])
        
        fnirs_data = self.fnirs_X[idx]
        label = self.y[idx]
        label_class = self.y_classes[idx]
        return torch.tensor(eeg_data, dtype=torch.float32), torch.tensor(fnirs_data, dtype=torch.float32), label, label_class

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

def train_model(model, dataloader, optimizer, device, precomputed_embeddings):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.90
    mse_loss_fn = nn.MSELoss()

    for batch_idx, (eeg_data, fnirs_data, labels, classes) in enumerate(dataloader):
        if not model.no_eeg:
            eeg_data = eeg_data.to(device)
        if not model.no_fnirs:
            fnirs_data = fnirs_data.to(device)
        
        # Get precomputed image embeddings for this batch
        img_features, valid_indices = get_batch_image_embeddings(labels, precomputed_embeddings, device)
        
        # Filter data to match valid labels
        if len(valid_indices) != len(labels):
            if not model.no_eeg:
                eeg_data = eeg_data[valid_indices]
            if not model.no_fnirs:
                fnirs_data = fnirs_data[valid_indices]
            labels = [labels[i] for i in valid_indices]
        
        if len(eeg_data) == 0:
            continue
            
        img_features = img_features.float()

        optimizer.zero_grad()

        batch_size = eeg_data.size(0) if not model.no_eeg else fnirs_data.size(0)
        subject_ids = torch.full((batch_size,), 1, dtype=torch.long).to(device)
        
        if model.no_eeg:
            # Create dummy EEG data for fNIRS-only model
            eeg_data = torch.zeros_like(fnirs_data)  # Use same shape as fNIRS for compatibility
            features = model(eeg_data, fnirs_data, subject_ids).float()
        elif model.no_fnirs:
            # Create dummy fNIRS data for EEG-only model
            fnirs_data = torch.zeros_like(eeg_data)  # Use same shape as EEG for compatibility
            features = model(eeg_data, fnirs_data, subject_ids).float()
        else:
            features = model(eeg_data, fnirs_data, subject_ids).float()

        logit_scale = model.logit_scale

        img_loss = model.loss_func(features, img_features, logit_scale)
        regress_loss = mse_loss_fn(features, img_features)
        loss = (alpha * regress_loss * 10 + (1 - alpha) * img_loss * 10)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        # Compute accuracy
        logits_img = logit_scale * features @ img_features.T
        predicted = torch.argmax(logits_img, dim=1)
        correct += (predicted == torch.arange(len(features)).to(device)).sum().item()
        total += len(features)

    average_loss = total_loss / (batch_idx + 1) if batch_idx >= 0 else 0
    accuracy = correct / total if total > 0 else 0
    return average_loss, accuracy

def evaluate_model(model, dataloader, device, precomputed_embeddings, valid_labels, k_values=[2, 4, 10, 50, 100]):
    model.eval()
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
            for eeg_data, fnirs_data, labels, classes in dataloader:
                if not model.no_eeg:
                    eeg_data = eeg_data.to(device)
                if not model.no_fnirs:
                    fnirs_data = fnirs_data.to(device)
                batch_size = eeg_data.size(0) if not model.no_eeg else fnirs_data.size(0)
                subject_ids = torch.full((batch_size,), 1, dtype=torch.long).to(device)
                
                if model.no_eeg:
                    # Create dummy EEG data for fNIRS-only model
                    eeg_data = torch.zeros_like(fnirs_data)  # Use same shape as fNIRS for compatibility
                    features = model(eeg_data, fnirs_data, subject_ids)
                elif model.no_fnirs:
                    # Create dummy fNIRS data for EEG-only model
                    fnirs_data = torch.zeros_like(eeg_data)  # Use same shape as EEG for compatibility
                    features = model(eeg_data, fnirs_data, subject_ids)
                else:
                    features = model(eeg_data, fnirs_data, subject_ids)
                
                logit_scale = model.logit_scale
                
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
                    logits = logit_scale * features[idx] @ selected_img_features.T
                    
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
    parser = argparse.ArgumentParser(description='ATMS Training for EEG-fNIRS-Image Dataset')
    parser.add_argument('--experiment_id', type=str, default='gtec_250527_data_250527_eeg_fnirs', help='Experiment ID')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--save_every', type=int, default=5, help='Save model every N epochs')
    parser.add_argument('--no_fnirs', action='store_true', help='Use EEG-only model')
    parser.add_argument('--no_eeg', action='store_true', help='Use fNIRS-only model')
    args = parser.parse_args()

    # Setup paths
    experiment_folder = f"./experiments/experiment_{args.experiment_id}"
    eeg_data_path = os.path.join(experiment_folder, 'whitened_eeg_data.npy') if not args.no_eeg else None
    fnirs_data_path = os.path.join(experiment_folder, 'whitened_fnirs_data.npy') if not args.no_fnirs else None
    stimuli_folder = os.path.join(experiment_folder, 'image_pool')
    
    # Create output directories
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    
    # Determine model name based on configuration
    if args.no_eeg and args.no_fnirs:
        raise ValueError("Both no_eeg and no_fnirs cannot be True simultaneously")
    elif args.no_eeg:
        model_name = "ATMS_fNIRS_only"
    elif args.no_fnirs:
        model_name = "ATMS_EEG_only"
    else:
        model_name = "ATMS_Multimodal"
    
    model_dir = os.path.join(experiment_folder, 'models', model_name, current_time)
    results_dir = os.path.join(experiment_folder, 'training_results', current_time)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine model type description
    if args.no_eeg:
        model_type = "fNIRS-only"
    elif args.no_fnirs:
        model_type = "EEG-only"
    else:
        model_type = "EEG-fNIRS multimodal"
    print(f"Model type: {model_type}")

    # Load datasets
    print('Loading datasets...')
    train_dataset = EEGfNIRSImageDataset(eeg_data_path, fnirs_data_path, train=True)
    test_dataset = EEGfNIRSImageDataset(eeg_data_path, fnirs_data_path, train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Precompute image embeddings
    print('Precomputing image embeddings...')
    all_labels = list(train_dataset.y) + list(test_dataset.y)
    precomputed_embeddings, valid_labels = precompute_image_embeddings(stimuli_folder, all_labels, device)

    # Initialize model
    print('Initializing ATMS model...')
    model = ATMS_Multimodal(
        eeg_channels=train_dataset.eeg_X.shape[1] if not args.no_eeg else 1,
        fnirs_channels=train_dataset.fnirs_X.shape[1] if not args.no_fnirs else 1,
        eeg_seq_len=train_dataset.eeg_X.shape[2] if not args.no_eeg else 1,
        fnirs_seq_len=train_dataset.fnirs_X.shape[2] if not args.no_fnirs else 1,
        no_fnirs=args.no_fnirs,
        no_eeg=args.no_eeg
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Training loop
    print('Starting training...')
    train_losses, train_accuracies = [], []
    test_results_history = []
    best_accuracy = 0.0
    best_epoch_info = {}

    for epoch in tqdm(range(args.epochs)):
        # Train
        train_loss, train_accuracy = train_model(model, train_loader, optimizer, device, precomputed_embeddings)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate
        test_results = evaluate_model(model, test_loader, device, precomputed_embeddings, valid_labels)
        test_results_history.append(test_results)

        # Save model periodically
        if (epoch + 1) % args.save_every == 0:
            model_path = os.path.join(model_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
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
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        for k, acc in test_results.items():
            print(f"Test {k}: {acc:.4f}")
        print("-" * 50)

    # Save final model
    final_model_path = os.path.join(model_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

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
        f.write(f"Model type: {model_type}\n")
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
    info_text += f"Model type: {model_type}\n"
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