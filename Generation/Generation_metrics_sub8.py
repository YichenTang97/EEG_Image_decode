import os
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from itertools import combinations
import clip
import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from eegdatasets_leaveone import EEGDataset
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from util import wandb_logger
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
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
from torch import nn
from torch.optim import AdamW
from diffusion_prior import *
from custom_pipeline import *
from PIL import Image
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable blocking for debugging

class Config:
    def __init__(self):
        self.task_name = 'classification'  # Example task name
        self.seq_len = 250                 # Sequence length
        self.pred_len = 250                # Prediction length
        self.output_attention = False      # Whether to output attention weights
        self.d_model = 250                 # Model dimension
        self.embed = 'timeF'               # Time encoding method
        self.freq = 'h'                    # Time frequency
        self.dropout = 0.25                # Dropout rate
        self.factor = 1                    # Attention scaling factor
        self.n_heads = 4                   # Number of attention heads
        self.e_layers = 1                  # Number of encoder layers
        self.d_ff = 256                    # Dimension of the feedforward network
        self.activation = 'gelu'           # Activation function
        self.enc_in = 63                   # Encoder input dimension (example value)

class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False,  num_subjects=10):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
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
        enc_out = enc_out[:, :63, :]      
        return enc_out

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
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
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
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
        self.encoder = iTransformer(default_config)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()       
         
    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        return out  

class EmbeddingDataset(Dataset):
    def __init__(self, c_embeddings=None, h_embeddings=None):
        self.c_embeddings = c_embeddings
        self.h_embeddings = h_embeddings

    def __len__(self):
        return len(self.c_embeddings)

    def __getitem__(self, idx):
        return {
            "c_embedding": self.c_embeddings[idx],
            "h_embedding": self.h_embeddings[idx]
        }

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def compute_eeg_embeddings(eeg_model, dataloader, device, sub):
    eeg_model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for eeg_data, _, _, _, img, _ in dataloader:
            eeg_data = eeg_data.to(device)
            subject_id = 1 # extract_id_from_string(sub)
            subject_ids = torch.full((eeg_data.size(0),), subject_id, dtype=torch.long).to(device)
            embeddings.append(eeg_model(eeg_data, subject_ids))
            labels.extend([os.path.splitext(os.path.basename(i))[0] for i in img])
    return torch.cat(embeddings, dim=0), labels

def evaluate_model(sub, eeg_model, dataloader, device, text_features_all, img_features_all, k):
    eeg_model.eval()

    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.99
    top5_correct = 0
    top5_correct_count = 0
    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0
    mse_loss_fn = nn.MSELoss()
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = eeg_data.size(0)  # Assume the first element is the data tensor
            subject_id = extract_id_from_string(sub)
            # eeg_data = eeg_data.permute(0, 2, 1)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
            # if not config.insubject:
            #     subject_ids = torch.full((batch_size,), -1, dtype=torch.long).to(device)          
            eeg_features = eeg_model(eeg_data, subject_ids)

        
            logit_scale = eeg_model.logit_scale 
            # print(eeg_features.type, text_features.type, img_features.type)
            img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
            text_loss = eeg_model.loss_func(eeg_features, text_features, logit_scale)
            regress_loss =  mse_loss_fn(eeg_features, img_features)
            loss = (alpha * regress_loss *10 + (1 - alpha) * img_loss*10)
                
            total_loss += loss.item()
            
            for idx, label in enumerate(labels):
                # First select k-1 classes excluding the correct class
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                selected_text_features = text_features_all[selected_classes]
                
                if k==200:
                    # Compute corresponding logits
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    # Get predicted class
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) ∈ {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        # print("predicted_label", predicted_label)
                        correct += 1
                    
                    # logits_single is the model output, assumed to be shape (n_batch, n_classes)
                    # label is the true label, shape (n_batch,)
                    # Get top-5 predicted indices
                    # print("logits_single", logits_single)
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                           
                    # Check if true label is in top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k == 50 or k == 100:
                    # For k=50 or 100, select k classes for evaluation
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]

                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    if predicted_label == label.item():
                        correct += 1
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                           
                    # Check if true label is in top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k==2 or k==4 or k==10:
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                    # Compute corresponding logits
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T
                    # logits_single = (logits_text + logits_img) / 2.0
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    # Get predicted class
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) ∈ {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        correct += 1
                    total += 1
                else:
                    print("Error.")
            del eeg_data, eeg_features, img_features
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total
    return average_loss, accuracy, top5_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Image Generation or Classification")
    parser.add_argument("--task", type=str, default="generate", choices=["generate", "retrival"], help="Task to perform: 'generate' or 'retrival'")
    args = parser.parse_args()

    assert args.task in ["generate", "retrival"], "Invalid task. Choose either 'generate' or 'retrival'."
    sub = "sub-08"
    data_path = "./../Preprocessed_data_250Hz"
    output_path = f"./fintune_ckpts/across/{sub}/"
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Loading EEG dataset...')
    train_dataset = EEGDataset(data_path, subjects=['sub-08'], train=True)
    test_dataset = EEGDataset(data_path, subjects=['sub-08'], train=False)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False)

    print('Loading EEG model...')
    eeg_model = ATMS(63, 250)
    eeg_model.load_state_dict(torch.load(f"models/contrast/across/exclude_sub-08/ATMS/05-11_09-48/40.pth"))
    eeg_model = eeg_model.to(device)

    print('Computing EEG embeddings...')
    emb_eeg_train, labels_train = compute_eeg_embeddings(eeg_model, train_loader, device, sub)
    emb_eeg_test, labels_test = compute_eeg_embeddings(eeg_model, test_loader, device, sub)

    print('Loading image embeddings...')
    emb_img_train = torch.load("variables/ViT-H-14_features_train.pt")
    emb_img_test = torch.load("variables/ViT-H-14_features_test.pt")
    emb_img_train_4 = emb_img_train.view(1654, 10, 1, 1024).repeat(1, 1, 4, 1).view(-1, 1024)

    print('Creating embedding dataset...')
    train_dataset = EmbeddingDataset(c_embeddings=emb_eeg_train, h_embeddings=emb_img_train_4)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0, pin_memory=False)

    print('Setting up diffusion prior...')
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1).to(device)
    pipe = Pipe(diffusion_prior, device=device)

    save_path = os.path.join(output_path, "diffusion_prior.pt")
    if os.path.exists(save_path):
        print('Loading existing diffusion prior model...')
        pipe.diffusion_prior.load_state_dict(torch.load(save_path))
    else:
        print('Training diffusion prior...')
        pipe.train(train_loader, num_epochs=150, learning_rate=1e-3)

        # Clear CUDA memory after training
        torch.cuda.empty_cache()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(pipe.diffusion_prior.state_dict(), save_path)

    if args.task == "generate":
        print('Generating images...')
        generator = Generator4Embeds(num_inference_steps=50, device=device)
        output_dir = f"./generated_imgs/across/{sub}/"
        os.makedirs(output_dir, exist_ok=True)

        for k in range(200):
            eeg_embeds = emb_eeg_test[k:k+1]
            label = labels_test[k]
            h = pipe.generate(c_embeds=eeg_embeds, num_inference_steps=50, guidance_scale=5.0)
            print(f"Generated embedding for label {label}")
            for j in range(10):
                image = generator.generate(h.to(dtype=torch.float16))
                image_path = os.path.join(output_dir, f"trial_{k}_label_{label}_gen_{j}.png")
                image.save(image_path)
                print(f"Image saved to {image_path}")
    elif args.task == "retrival":
        output_dir = f"./outputs/contrast/"
        os.makedirs(output_dir, exist_ok=True)
        # Evaluate the model
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
        text_features_test_all = test_dataset.text_features
        img_features_test_all = test_dataset.img_features
        test_loss, test_accuracy, top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all,k=200)
        _, v2_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 2)
        _, v4_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 4)
        _, v10_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 10)
        _, v50_acc, v50_top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all,  k=50)
        _, v100_acc, v100_top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all,  k=100)

        # Save results to a CSV file
        current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
        results_dir = os.path.join(output_dir, 'ATMS', sub, current_time)
        os.makedirs(results_dir, exist_ok=True)

        results_file = os.path.join(results_dir, "retrieval_scores.csv")
        with open(results_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Test Loss", test_loss])
            writer.writerow(["Test Accuracy", test_accuracy])
            writer.writerow(["Top-5 Accuracy", top5_acc])
            writer.writerow(["k=2 Accuracy", v2_acc])
            writer.writerow(["k=4 Accuracy", v4_acc])
            writer.writerow(["k=10 Accuracy", v10_acc])
            writer.writerow(["k=50 Accuracy", v50_acc])
            writer.writerow(["k=50 Top-5 Accuracy", v50_top5_acc])
            writer.writerow(["k=100 Accuracy", v100_acc])
            writer.writerow(["k=100 Top-5 Accuracy", v100_top5_acc])

        print(f"Retrieval scores saved to {results_file}")