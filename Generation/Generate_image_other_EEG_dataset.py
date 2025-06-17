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
import argparse
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
from collections import defaultdict
import datetime

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
    parser = argparse.ArgumentParser(description="EEG Image Generation or Classification")
    parser.add_argument("--task", type=str, default="generate", choices=["generate", "classify"], help="Task to perform: 'generate' or 'classify'")
    parser.add_argument("--classify_with_prior", action="store_true", help="Use diffusion prior for classification")
    parser.add_argument("--generate_without_prior", action="store_true", help="Generate images directly from EEG embeddings without diffusion prior")
    parser.add_argument("--regenerate_stimuli", action="store_true", help="Regenerate stimuli images")
    parser.add_argument("--generation_repeats", type=int, default=3, help="Number of images to generate per trial")
    parser.add_argument("--guidance_scale", type=float, default=50.0, help="Guidance scale for diffusion prior generation")
    args = parser.parse_args()

    assert args.task in ["generate", "classify"], "Invalid task. Choose either 'generate' or 'classify'."

    experiment_id = "biosemi_imagery_250414_data_250509"  # Configurable experiment ID
    experiment_folder = f"./experiments/experiment_{experiment_id}"

    data_path = os.path.join(experiment_folder, 'whitened_eeg_data_for_li_ATM_reconstruct.npy')
    stimuli_folder = os.path.join(experiment_folder, 'stimuli_shapes')
    diffusion_prior_folder = os.path.join(experiment_folder, 'diffusion_prior')
    
    # Create timestamped folder for generated images
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.task == "generate":
        if args.generate_without_prior:
            generated_images_folder = os.path.join(experiment_folder, f'generated_images_without_prior_{timestamp}')
        else:
            generated_images_folder = os.path.join(experiment_folder, f'generated_images_with_prior_{timestamp}')
    else:
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
    eeg_model.load_state_dict(torch.load(f"./models/contrast/across/ATMS/05-08_19-24/40.pth")) # Biosemi
    # eeg_model.load_state_dict(torch.load(f"models/contrast/across/ATMS/05-08_17-41/40.pth")) # mbt
    eeg_model = eeg_model.to(device)

    print('Computing EEG embeddings...')
    emb_eeg_train, labels_train = compute_eeg_embeddings_other(eeg_model, train_loader, device)
    emb_eeg_test, labels_test = compute_eeg_embeddings_other(eeg_model, test_loader, device)

    emb_img_train = compute_image_embeddings(stimuli_folder, labels_train, device)

    if args.regenerate_stimuli:
        print('Regenerating images for each unique class...')

        # Compute embeddings for unique classes
        unique_labels = list(set(labels_train))
        emb_img_classes = compute_image_embeddings(stimuli_folder, unique_labels, device)

        # Create timestamped subfolder for regenerated stimuli
        regenerated_stimuli_folder = os.path.join(experiment_folder, f'regenerated_stimuli_{timestamp}')
        os.makedirs(regenerated_stimuli_folder, exist_ok=True)

        # Regenerate images for each class
        generator = Generator4Embeds(num_inference_steps=50, device=device)
        for class_idx, class_embed in enumerate(emb_img_classes):
            for j in range(args.generation_repeats):
                image = generator.generate(class_embed.unsqueeze(0).to(dtype=torch.float16))
                image_path = os.path.join(regenerated_stimuli_folder, f"class_{unique_labels[class_idx]}_gen_{j}.png")
                image.save(image_path)
                print(f"Regenerated image saved to {image_path}")

    # Only setup diffusion prior if needed
    diffusion_prior = None
    pipe = None
    if not args.generate_without_prior or args.classify_with_prior:
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
    else:
        print('Skipping diffusion prior setup (generating without prior)...')

    if args.task == "generate":
        print('Generating images...')
        generator = Generator4Embeds(num_inference_steps=50, device=device)

        for k in range(len(emb_eeg_test)):
            eeg_embeds = emb_eeg_test[k:k+1]
            label = labels_test[k]
            
            if args.generate_without_prior:
                # Generate directly from EEG embeddings
                print(f"Generating image {k+1}/{len(emb_eeg_test)} without diffusion prior...")
                h = eeg_embeds.squeeze(0)  # Use EEG embedding directly
                # Note: There might be dimension mismatches since EEG embeddings are 1024D 
                # and the generator expects image embeddings which might have different dimensions
                try:
                    for j in range(args.generation_repeats):
                        image = generator.generate(h.unsqueeze(0).to(dtype=torch.float16))
                        image_path = os.path.join(generated_images_folder, f"trial_{k}_label_{label}_gen_{j}_no_prior.png")
                        image.save(image_path)
                        print(f"Image saved to {image_path}")
                except Exception as e:
                    print(f"Warning: Failed to generate image without prior for trial {k}: {e}")
                    print("This might be due to dimension mismatches between EEG and image embeddings.")
                    continue
            else:
                # Generate using diffusion prior (original method)
                print(f"Generating image {k+1}/{len(emb_eeg_test)} with diffusion prior...")
                h = pipe.generate(c_embeds=eeg_embeds, num_inference_steps=50, guidance_scale=args.guidance_scale)
                for j in range(args.generation_repeats):
                    image = generator.generate(h.to(dtype=torch.float16))
                    image_path = os.path.join(generated_images_folder, f"trial_{k}_label_{label}_gen_{j}.png")
                    image.save(image_path)
                    print(f"Image saved to {image_path}")
    elif args.task == "classify":
        print('Computing image embeddings for all classes...')
        unique_train_labels = list(set(labels_train))
        emb_img_classes = compute_image_embeddings(stimuli_folder, unique_train_labels, device)

        print('Classifying EEG embeddings...')
        predictions = []
        scores = []
        for idx, eeg_embed in enumerate(emb_eeg_test):
            if args.classify_with_prior:
                # Use diffusion prior for classification
                h = pipe.generate(c_embeds=eeg_embed.unsqueeze(0), num_inference_steps=50, guidance_scale=args.guidance_scale).squeeze(0)
                h = h.to(dtype=torch.float16)
            else:
                # Use EEG embedding directly
                h = eeg_embed
            # Compute logits for classification
            logit_scale = eeg_model.logit_scale
            logits_img = logit_scale * eeg_embed @ emb_img_classes.T
            logits_single = logits_img

            # Predict the class
            predicted_class_idx = torch.argmax(logits_single).item()
            predicted_class = unique_train_labels[predicted_class_idx]
            predictions.append(predicted_class)
            scores.append(logits_single.cpu().detach().numpy())

        print('Evaluating classification results...')
        y_true = labels_test
        y_pred = predictions

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_train_labels)

        print('Logging results...')
        if args.classify_with_prior:
            results_path = os.path.join(experiment_folder, "classification_results_with_prior.txt")
        else:
            results_path = os.path.join(experiment_folder, "classification_results.txt")
        with open(results_path, "w") as f:
            f.write(f"Accuracy: {acc}\n")
            f.write(f"Balanced Accuracy: {bal_acc}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write(f"MCC: {mcc}\n")
            f.write(f"Confusion Matrix:\n{conf_matrix}\n")
            f.write(f"Class/Label Order: {unique_train_labels}\n")

        print('Computing class-wise scores...')
        class_wise_results = defaultdict(dict)
        for class_label in set(y_true):
            class_indices = [i for i, label in enumerate(y_true) if label == class_label]
            y_true_class = [y_true[i] for i in class_indices]
            y_pred_class = [y_pred[i] for i in class_indices]

            class_acc = accuracy_score(y_true_class, y_pred_class)
            class_f1 = f1_score(y_true_class, y_pred_class, average='weighted')
            class_mcc = matthews_corrcoef(y_true_class, y_pred_class)

            class_wise_results[class_label]['accuracy'] = class_acc
            class_wise_results[class_label]['f1_score'] = class_f1
            class_wise_results[class_label]['mcc'] = class_mcc

        print('Logging class-wise results...')
        if args.classify_with_prior:
            class_wise_results_path = os.path.join(experiment_folder, "class_wise_results_with_prior.txt")
        else:
            class_wise_results_path = os.path.join(experiment_folder, "class_wise_results.txt")
        with open(class_wise_results_path, "w") as f:
            for class_label, metrics in class_wise_results.items():
                f.write(f"Class {class_label}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']}\n")
                f.write(f"  F1 Score: {metrics['f1_score']}\n")
                f.write(f"  MCC: {metrics['mcc']}\n")
                f.write("\n")
        print(f"Class-wise results saved to {class_wise_results_path}")

        print('Saving predictions and scores...')
        if args.classify_with_prior:
            predictions_scores_path = os.path.join(experiment_folder, "predictions_and_scores_with_prior.npy")
        else:
            predictions_scores_path = os.path.join(experiment_folder, "predictions_and_scores.npy")
        np.save(predictions_scores_path, {"predictions": predictions, "scores": scores})
        print(f"Results saved to {results_path} and {predictions_scores_path}")

if __name__ == "__main__":
    main()