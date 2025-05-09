import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ATMS_reconstruction_cross_subjects import ATMS
from diffusion_prior import DiffusionPriorUNet, Pipe
from custom_pipeline import Generator4Embeds
from PIL import Image
from torchvision import transforms
import open_clip
import einops

class EEGDatasetOther(Dataset):
    def __init__(self, data_path, train=True):
        self.data_path = data_path
        self.train = train
        self.data = np.load(data_path, allow_pickle=True).item()
        self.X = self.data['X_train'] if train else self.data['X_test']
        self.y = self.data['y_train'] if train else self.data['y_test']

        if train:
            n_trials, n_repeats, n_channels, n_timepoints = self.X.shape
            self.X = einops.rearrange(self.X, 'n r c t -> (n r) c t')
            self.y = einops.repeat(self.y, 'n -> (n r)', r=n_repeats)
        else:
            self.X = np.mean(self.X, axis=1)
        
        print(f"EEG data shape: {self.X.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        eeg_data = self.X[idx]
        label = self.y[idx]
        return torch.tensor(eeg_data, dtype=torch.float32), label

class EmbeddingDataset(Dataset):
    def __init__(self, c_embeddings, h_embeddings):
        self.c_embeddings = c_embeddings
        self.h_embeddings = h_embeddings

    def __len__(self):
        return len(self.c_embeddings)

    def __getitem__(self, idx):
        return {
            "c_embedding": self.c_embeddings[idx],
            "h_embedding": self.h_embeddings[idx]
        }

def compute_image_embeddings(stimuli_folder, labels, device):
    print('Computing image embeddings...')
    model, preprocess, _ = open_clip.create_model_and_transforms('ViT-H-14', 
            pretrained="./variables/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin", 
            precision='fp32', device=device)

    # Map label to image path
    label_to_image = {img.split('_')[1].split('.')[0]: os.path.join(stimuli_folder, img) for img in os.listdir(stimuli_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))}

    # Compute unique image embeddings
    unique_labels = set(labels)
    unique_image_features = {}
    for label in unique_labels:
        image_path = label_to_image[label]
        image_input = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
        unique_image_features[label] = image_feature

    # Reorder and concatenate embeddings based on labels
    image_features_list = [unique_image_features[label].clone().detach() for label in labels]
    return torch.cat(image_features_list, dim=0)

def compute_eeg_embeddings_other(eeg_model, dataloader, device):
    eeg_model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for eeg_data, label in dataloader:
            eeg_data = eeg_data.to(device)
            subject_id = 1
            subject_ids = torch.full((eeg_data.size(0),), subject_id, dtype=torch.long).to(device)
            embeddings.append(eeg_model(eeg_data, subject_ids))
            labels.extend(label)
    return torch.cat(embeddings, dim=0), labels

def main():
    experiment_id = "mbt_250411_data_250509"  # Configurable experiment ID
    experiment_folder = f"./experiment_{experiment_id}"

    data_path = os.path.join(experiment_folder, 'whitened_eeg_data_for_li_ATM_reconstruct.npy')
    stimuli_folder = os.path.join(experiment_folder, 'stimuli_shapes')
    diffusion_prior_folder = os.path.join(experiment_folder, 'diffusion_prior')
    generated_images_folder = os.path.join(experiment_folder, 'generated_images')

    os.makedirs(diffusion_prior_folder, exist_ok=True)
    os.makedirs(generated_images_folder, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Loading EEG dataset...')
    train_dataset = EEGDatasetOther(data_path, train=True)
    test_dataset = EEGDatasetOther(data_path, train=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    print('Loading EEG model...')
    eeg_model = ATMS(num_channels=train_dataset.X.shape[1], sequence_length=250)
    # eeg_model.load_state_dict(torch.load(f"./models/contrast/across/ATMS/05-08_19-24/40.pth")) # Biosemi
    eeg_model.load_state_dict(torch.load(f"models/contrast/across/ATMS/05-08_17-41/40.pth")) # mbt
    eeg_model = eeg_model.to(device)

    print('Computing EEG embeddings...')
    emb_eeg_train, labels_train = compute_eeg_embeddings_other(eeg_model, train_loader, device)
    emb_eeg_test, labels_test = compute_eeg_embeddings_other(eeg_model, test_loader, device)

    emb_img_train = compute_image_embeddings(stimuli_folder, labels_train, device)

    print('Creating embedding dataset...')
    embedding_dataset = EmbeddingDataset(c_embeddings=emb_eeg_train, h_embeddings=emb_img_train)
    embedding_loader = DataLoader(embedding_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)

    print('Setting up diffusion prior...')
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1).to(device)
    pipe = Pipe(diffusion_prior, device=device)

    save_path = os.path.join(diffusion_prior_folder, "diffusion_prior.pt")
    if os.path.exists(save_path):
        print('Loading existing diffusion prior model...')
        pipe.diffusion_prior.load_state_dict(torch.load(save_path))
    else:
        print('Training diffusion prior...')
        pipe.train(embedding_loader, num_epochs=150, learning_rate=1e-3)
        torch.save(pipe.diffusion_prior.state_dict(), save_path)

    print('Generating images...')
    generator = Generator4Embeds(num_inference_steps=50, device=device)

    for k in range(len(emb_eeg_test)):
        eeg_embeds = emb_eeg_test[k:k+1]
        label = labels_test[k]
        h = pipe.generate(c_embeds=eeg_embeds, num_inference_steps=50, guidance_scale=5.0)
        for j in range(10):
            image = generator.generate(h.to(dtype=torch.float16))
            image_path = os.path.join(generated_images_folder, f"trial_{k}_label_{label}_gen_{j}.png")
            image.save(image_path)
            print(f"Image saved to {image_path}")

if __name__ == "__main__":
    main()