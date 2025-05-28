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
import random
import csv
import datetime
import torch.nn as nn

class EEGDatasetCoco(Dataset):
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
    label_to_image = {img.split('.')[0]: os.path.join(stimuli_folder, img) for img in os.listdir(stimuli_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))}

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
    return torch.cat(image_features_list, dim=0).to(dtype=torch.float32)

def compute_eeg_embeddings_other(eeg_model, dataloader, device):
    eeg_model.eval()
    embeddings = []
    labels = []
    classes = []
    with torch.no_grad():
        for eeg_data, label, c in dataloader:
            eeg_data = eeg_data.to(device)
            subject_id = 1
            subject_ids = torch.full((eeg_data.size(0),), subject_id, dtype=torch.long).to(device)
            embeddings.append(eeg_model(eeg_data, subject_ids))
            labels.extend(label)
            classes.extend(c)
    return torch.cat(embeddings, dim=0).to(dtype=torch.float32), labels, classes

def evaluate_model(eeg_model, eeg_embeddings, labels, classes, img_embeddings, unique_labels, device, k, use_prior=False, pipe=None):
    """
    Evaluate model performance for image retrieval and classification.
    
    Args:
        eeg_model: The EEG model
        eeg_embeddings: EEG embeddings tensor
        labels: List of image labels
        classes: List of class labels
        img_embeddings: Image embeddings tensor for unique labels
        unique_labels: List of unique labels corresponding to img_embeddings
        device: Device to run on
        k: Number of classes to consider for evaluation
        use_prior: Whether to use diffusion prior
        pipe: Diffusion prior pipeline (required if use_prior=True)
    
    Returns:
        accuracy, top5_accuracy, class_accuracy, predictions_dict
    """
    eeg_model.eval()
    
    correct = 0
    total = 0
    top5_correct_count = 0
    class_correct = 0
    
    # Storage for predictions
    predictions_dict = {
        'predicted_labels': [],
        'predicted_classes': [],
        'true_labels': [],
        'true_classes': [],
        'logits': [],
        'top5_predictions': []
    }
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Create class to label mapping
    class_to_labels = defaultdict(list)
    for label, cls in zip(labels, classes):
        if label in label_to_idx:
            class_to_labels[cls].append(label)
    
    all_label_indices = set(range(len(unique_labels)))
    
    with torch.no_grad():
        for idx, (eeg_embed, label, cls) in enumerate(zip(eeg_embeddings, labels, classes)):
            if label not in label_to_idx:
                continue
                
            true_label_idx = label_to_idx[label]
            
            # Generate embedding using diffusion prior if specified
            if use_prior and pipe is not None:
                h = pipe.generate(c_embeds=eeg_embed.unsqueeze(0), num_inference_steps=50, guidance_scale=5.0).squeeze(0)
                query_embed = h.to(dtype=torch.float32)
            else:
                query_embed = eeg_embed.to(dtype=torch.float32)
            
            # Select k classes for evaluation
            if k >= len(unique_labels):
                # Use all available classes
                selected_indices = list(all_label_indices)
                selected_labels = unique_labels
            else:
                # Select k-1 random classes excluding the correct class
                possible_indices = list(all_label_indices - {true_label_idx})
                selected_indices = random.sample(possible_indices, min(k-1, len(possible_indices))) + [true_label_idx]
                selected_labels = [unique_labels[i] for i in selected_indices]
            
            selected_img_features = img_embeddings[selected_indices].to(dtype=torch.float32)
            
            # Compute logits for retrieval
            logit_scale = eeg_model.logit_scale.to(dtype=torch.float32)
            logits = logit_scale * query_embed @ selected_img_features.T
            
            # Image retrieval accuracy
            predicted_idx = torch.argmax(logits).item()
            predicted_label = selected_labels[predicted_idx]
            
            if predicted_label == label:
                correct += 1
            
            # Top-5 accuracy (if k >= 5)
            top5_pred_labels = []
            if k >= 5:
                _, top5_indices = torch.topk(logits, min(5, len(selected_indices)), largest=True)
                top5_pred_labels = [selected_labels[i] for i in top5_indices.tolist()]
                if true_label_idx in [selected_indices[i] for i in top5_indices.tolist()]:
                    top5_correct_count += 1
            
            # Class-level accuracy
            predicted_class = predicted_label.split('_')[0] if '_' in predicted_label else predicted_label
            true_class = cls
            
            if predicted_class == true_class:
                class_correct += 1
            
            # Store predictions
            predictions_dict['predicted_labels'].append(predicted_label)
            predictions_dict['predicted_classes'].append(predicted_class)
            predictions_dict['true_labels'].append(label)
            predictions_dict['true_classes'].append(true_class)
            predictions_dict['logits'].append(logits.cpu().detach().numpy())
            predictions_dict['top5_predictions'].append(top5_pred_labels)
            
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    top5_accuracy = top5_correct_count / total if total > 0 and k >= 5 else 0
    class_accuracy = class_correct / total if total > 0 else 0
    
    return accuracy, top5_accuracy, class_accuracy, predictions_dict

def main():
    """
    Main function for EEG-based image generation and retrieval evaluation.
    
    Folder Structure Created:
    experiment_{experiment_id}/
    ├── diffusion_prior/
    │   └── diffusion_prior.pt
    ├── generated_images/
    │   ├── {timestamp}_with_prior/
    │   │   └── label_{label}_gen_{j}.png
    │   └── {timestamp}_without_prior/
    │       └── label_{label}_gen_{j}.png
    ├── regenerated_stimuli/
    │   └── {timestamp}/
    │       └── {class}_gen_{j}.png
    └── retrieval_results/
        └── {timestamp}/
            ├── comprehensive_retrieval_results.csv
            ├── detailed_results.txt
            └── comprehensive_predictions.npy
    """
    parser = argparse.ArgumentParser(description="EEG Image Generation or Retrieval")
    parser.add_argument("--task", type=str, default="generate", choices=["generate", "retrival"], help="Task to perform: 'generate' or 'retrival'")
    parser.add_argument("--retrival_with_prior", action="store_true", help="Use diffusion prior for image retrieval")
    parser.add_argument("--generate_without_prior", action="store_true", help="Generate images without diffusion prior")
    parser.add_argument("--regenerate_stimuli", action="store_true", help="Regenerate stimuli images for test set")
    parser.add_argument("--generation_repeats", type=int, default=10, help="How many times to generate images for each embedding")
    args = parser.parse_args()

    assert args.task in ["generate", "retrival"], "Invalid task. Choose either 'generate' or 'retrival'."
    
    # Validate argument combinations
    if args.generate_without_prior and args.task != "generate":
        raise ValueError("--generate_without_prior can only be used with --task generate")

    experiment_id = "gtec_250527_data_250527"  # Configurable experiment ID
    experiment_folder = f"./experiments/experiment_{experiment_id}"

    data_path = os.path.join(experiment_folder, 'whitened_eeg_data_for_li_ATM_reconstruct.npy')
    stimuli_folder = os.path.join(experiment_folder, 'coco_image_pool')
    diffusion_prior_folder = os.path.join(experiment_folder, 'diffusion_prior')
    generated_images_folder = os.path.join(experiment_folder, 'generated_images')

    os.makedirs(diffusion_prior_folder, exist_ok=True)
    os.makedirs(generated_images_folder, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Loading EEG dataset...')
    train_dataset = EEGDatasetCoco(data_path, train=True)
    test_dataset = EEGDatasetCoco(data_path, train=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    print('Loading EEG model...')
    eeg_model = ATMS(num_channels=train_dataset.X.shape[1], sequence_length=250)
    eeg_model.load_state_dict(torch.load(f"./models/contrast/across/ATMS/05-27_13-46/40.pth"))
    eeg_model = eeg_model.to(device)

    print('Computing EEG embeddings...')
    emb_eeg_train, labels_train, classes_train = compute_eeg_embeddings_other(eeg_model, train_loader, device)
    emb_eeg_test, labels_test, classes_test = compute_eeg_embeddings_other(eeg_model, test_loader, device)

    emb_img_train = compute_image_embeddings(stimuli_folder, labels_train, device)

    gen_repeats = args.generation_repeats
    if args.regenerate_stimuli:
        print('Regenerating images for each unique test image...')

        # Compute embeddings for unique classes
        unique_labels = sorted(list(set(labels_test)))  # Ensure deterministic order
        emb_img_classes = compute_image_embeddings(stimuli_folder, unique_labels, device)

        # Create timestamped subfolder for regenerated stimuli
        current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
        regenerated_stimuli_folder = os.path.join(experiment_folder, 'regenerated_stimuli', current_time)
        os.makedirs(regenerated_stimuli_folder, exist_ok=True)

        # Regenerate images for each class
        generator = Generator4Embeds(num_inference_steps=50, device=device)
        for idx, embed in enumerate(emb_img_classes):
            for j in range(gen_repeats):
                image = generator.generate(embed.unsqueeze(0).to(dtype=torch.float16))
                image_path = os.path.join(regenerated_stimuli_folder, f"{unique_labels[idx]}_gen_{j}.png")
                image.save(image_path)
                print(f"Regenerated image saved to {image_path}")
        
        print(f"All regenerated stimuli saved to {regenerated_stimuli_folder}")
        return

    print('Creating embedding dataset...')
    embedding_dataset = EmbeddingDataset(c_embeddings=emb_eeg_train, h_embeddings=emb_img_train)
    embedding_loader = DataLoader(embedding_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)

    # Setup diffusion prior only if needed
    pipe = None
    if (args.task == "generate" and not args.generate_without_prior) or (args.task == "retrival"):
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
        print('Skipping diffusion prior setup...')

    if args.task == "generate":
        print('Generating images...')
        generator = Generator4Embeds(num_inference_steps=50, device=device)

        # Create timestamped subfolder for generated images
        current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
        method_suffix = "_without_prior" if args.generate_without_prior else "_with_prior"
        generated_images_timestamped_folder = os.path.join(generated_images_folder, current_time + method_suffix)
        os.makedirs(generated_images_timestamped_folder, exist_ok=True)

        for k in range(len(emb_eeg_test)):
            eeg_embeds = emb_eeg_test[k:k+1]
            label = labels_test[k]
            
            if args.generate_without_prior:
                # Generate directly from EEG embeddings without diffusion prior
                h = eeg_embeds.to(dtype=torch.float16)
            else:
                # Use diffusion prior to generate intermediate embeddings
                h = pipe.generate(c_embeds=eeg_embeds, num_inference_steps=50, guidance_scale=5.0)
            
            for j in range(gen_repeats):
                image = generator.generate(h.to(dtype=torch.float16))
                image_path = os.path.join(generated_images_timestamped_folder, f"label_{label}_gen_{j}.png")
                image.save(image_path)
                print(f"Image saved to {image_path}")
        
        print(f"All generated images saved to {generated_images_timestamped_folder}")
        
        # Summary of generated content
        print("\n" + "="*60)
        print("GENERATION SUMMARY")
        print("="*60)
        print(f"Generation method: {'Without diffusion prior' if args.generate_without_prior else 'With diffusion prior'}")
        print(f"Total EEG samples processed: {len(emb_eeg_test)}")
        print(f"Images generated per sample: {gen_repeats}")
        print(f"Total images generated: {len(emb_eeg_test) * gen_repeats}")
        print(f"Generated images saved to: {generated_images_timestamped_folder}")
        if args.regenerate_stimuli:
            print(f"Regenerated stimuli saved to: {regenerated_stimuli_folder}")
        print("="*60)
    elif args.task == "retrival":
        print('Computing image embeddings for all images...')
        unique_test_labels = sorted(list(set(labels_test)))  # Ensure deterministic order
        emb_img_classes = compute_image_embeddings(stimuli_folder, unique_test_labels, device)

        print('Performing comprehensive retrieval and classification evaluation...')
        
        # Evaluation configurations
        k_values = [2, 4, 10, 50, 100, len(unique_test_labels)]
        
        # Results storage
        results = {}
        all_predictions = {}
        
        # Evaluate without diffusion prior
        print('Evaluating without diffusion prior...')
        for k in k_values:
            if k > len(unique_test_labels):
                k = len(unique_test_labels)
            
            acc, top5_acc, class_acc, predictions_dict = evaluate_model(
                eeg_model, emb_eeg_test, labels_test, classes_test, 
                emb_img_classes, unique_test_labels, device, k, 
                use_prior=False, pipe=None
            )
            
            results[f'k={k}_retrieval_acc'] = acc
            results[f'k={k}_class_acc'] = class_acc
            if k >= 5:
                results[f'k={k}_top5_acc'] = top5_acc
            
            # Store predictions for the full dataset evaluation (largest k)
            if k == len(unique_test_labels):
                all_predictions['no_prior'] = predictions_dict
            
            print(f"k={k}: Retrieval Acc={acc:.4f}, Class Acc={class_acc:.4f}, Top5 Acc={top5_acc:.4f}")
        
        # Evaluate with diffusion prior if enabled
        if args.retrival_with_prior:
            print('Evaluating with diffusion prior...')
            for k in k_values:
                if k > len(unique_test_labels):
                    k = len(unique_test_labels)
                
                acc, top5_acc, class_acc, predictions_dict = evaluate_model(
                    eeg_model, emb_eeg_test, labels_test, classes_test, 
                    emb_img_classes, unique_test_labels, device, k, 
                    use_prior=True, pipe=pipe
                )
                
                results[f'k={k}_retrieval_acc_with_prior'] = acc
                results[f'k={k}_class_acc_with_prior'] = class_acc
                if k >= 5:
                    results[f'k={k}_top5_acc_with_prior'] = top5_acc
                
                # Store predictions for the full dataset evaluation (largest k)
                if k == len(unique_test_labels):
                    all_predictions['with_prior'] = predictions_dict
                
                print(f"k={k} (with prior): Retrieval Acc={acc:.4f}, Class Acc={class_acc:.4f}, Top5 Acc={top5_acc:.4f}")
        
        # Save comprehensive results
        print('Saving comprehensive results...')
        current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
        results_dir = os.path.join(experiment_folder, 'retrieval_results', current_time)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save to CSV
        csv_path = os.path.join(results_dir, "comprehensive_retrieval_results.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            for metric, value in results.items():
                writer.writerow([metric, value])
        
        # Save detailed results to text file
        txt_path = os.path.join(results_dir, "detailed_results.txt")
        with open(txt_path, "w") as f:
            f.write("Comprehensive Retrieval and Classification Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Image Retrieval Accuracies (without prior):\n")
            for k in k_values:
                if k > len(unique_test_labels):
                    k = len(unique_test_labels)
                if f'k={k}_retrieval_acc' in results:
                    f.write(f"  k={k}: {results[f'k={k}_retrieval_acc']:.4f}\n")
            
            f.write("\nClass Classification Accuracies (without prior):\n")
            for k in k_values:
                if k > len(unique_test_labels):
                    k = len(unique_test_labels)
                if f'k={k}_class_acc' in results:
                    f.write(f"  k={k}: {results[f'k={k}_class_acc']:.4f}\n")
            
            f.write("\nTop-5 Accuracies (without prior):\n")
            for k in k_values:
                if k > len(unique_test_labels):
                    k = len(unique_test_labels)
                if f'k={k}_top5_acc' in results:
                    f.write(f"  k={k}: {results[f'k={k}_top5_acc']:.4f}\n")
            
            if args.retrival_with_prior:
                f.write("\nImage Retrieval Accuracies (with prior):\n")
                for k in k_values:
                    if k > len(unique_test_labels):
                        k = len(unique_test_labels)
                    if f'k={k}_retrieval_acc_with_prior' in results:
                        f.write(f"  k={k}: {results[f'k={k}_retrieval_acc_with_prior']:.4f}\n")
                
                f.write("\nClass Classification Accuracies (with prior):\n")
                for k in k_values:
                    if k > len(unique_test_labels):
                        k = len(unique_test_labels)
                    if f'k={k}_class_acc_with_prior' in results:
                        f.write(f"  k={k}: {results[f'k={k}_class_acc_with_prior']:.4f}\n")
                
                f.write("\nTop-5 Accuracies (with prior):\n")
                for k in k_values:
                    if k > len(unique_test_labels):
                        k = len(unique_test_labels)
                    if f'k={k}_top5_acc_with_prior' in results:
                        f.write(f"  k={k}: {results[f'k={k}_top5_acc_with_prior']:.4f}\n")
            
            f.write(f"\nTotal test samples: {len(labels_test)}\n")
            f.write(f"Unique labels: {len(unique_test_labels)}\n")
            f.write(f"Unique classes: {len(set(classes_test))}\n")
        
        print(f"Results saved to {csv_path} and {txt_path}")
        
        # Save comprehensive predictions for detailed analysis
        print('Saving comprehensive predictions for detailed analysis...')
        
        # Prepare comprehensive predictions data
        predictions_data = {
            "unique_test_labels": unique_test_labels,
            "total_test_samples": len(labels_test),
            "unique_classes": list(set(classes_test))
        }
        
        # Add predictions without prior
        if 'no_prior' in all_predictions:
            predictions_data.update({
                "predicted_labels_no_prior": all_predictions['no_prior']['predicted_labels'],
                "predicted_classes_no_prior": all_predictions['no_prior']['predicted_classes'],
                "true_labels": all_predictions['no_prior']['true_labels'],
                "true_classes": all_predictions['no_prior']['true_classes'],
                "logits_no_prior": all_predictions['no_prior']['logits'],
                "top5_predictions_no_prior": all_predictions['no_prior']['top5_predictions']
            })
        
        # Add predictions with prior if available
        if 'with_prior' in all_predictions:
            predictions_data.update({
                "predicted_labels_with_prior": all_predictions['with_prior']['predicted_labels'],
                "predicted_classes_with_prior": all_predictions['with_prior']['predicted_classes'],
                "logits_with_prior": all_predictions['with_prior']['logits'],
                "top5_predictions_with_prior": all_predictions['with_prior']['top5_predictions']
            })
        
        # Save comprehensive predictions
        predictions_path = os.path.join(results_dir, "comprehensive_predictions.npy")
        np.save(predictions_path, predictions_data)
        
        print(f"Comprehensive predictions saved to {predictions_path}")
        print("Comprehensive evaluation completed!")
        
        # Summary of retrieval evaluation
        print("\n" + "="*60)
        print("RETRIEVAL EVALUATION SUMMARY")
        print("="*60)
        print(f"Total test samples evaluated: {len(labels_test)}")
        print(f"Unique image labels: {len(unique_test_labels)}")
        print(f"Unique classes: {len(set(classes_test))}")
        print(f"K-values evaluated: {k_values}")
        print(f"Diffusion prior used: {'Yes' if args.retrival_with_prior else 'No'}")
        print(f"Results saved to: {results_dir}")
        print(f"CSV results: {csv_path}")
        print(f"Detailed results: {txt_path}")
        print(f"Predictions data: {predictions_path}")
        print("="*60)

if __name__ == "__main__":
    main()

def load_and_analyze_predictions(predictions_path):
    """
    Helper function to load and analyze saved predictions.
    
    Args:
        predictions_path: Path to the comprehensive_predictions.npy file
    
    Returns:
        Dictionary with loaded predictions and basic analysis
    """
    data = np.load(predictions_path, allow_pickle=True).item()
    
    analysis = {
        'data': data,
        'summary': {}
    }
    
    # Basic statistics
    if 'true_labels' in data:
        analysis['summary']['total_samples'] = len(data['true_labels'])
        analysis['summary']['unique_true_labels'] = len(set(data['true_labels']))
        analysis['summary']['unique_true_classes'] = len(set(data['true_classes']))
    
    # Accuracy analysis without prior
    if 'predicted_labels_no_prior' in data and 'true_labels' in data:
        correct_labels = sum(1 for pred, true in zip(data['predicted_labels_no_prior'], data['true_labels']) if pred == true)
        correct_classes = sum(1 for pred, true in zip(data['predicted_classes_no_prior'], data['true_classes']) if pred == true)
        
        analysis['summary']['label_accuracy_no_prior'] = correct_labels / len(data['true_labels'])
        analysis['summary']['class_accuracy_no_prior'] = correct_classes / len(data['true_classes'])
    
    # Accuracy analysis with prior
    if 'predicted_labels_with_prior' in data and 'true_labels' in data:
        correct_labels = sum(1 for pred, true in zip(data['predicted_labels_with_prior'], data['true_labels']) if pred == true)
        correct_classes = sum(1 for pred, true in zip(data['predicted_classes_with_prior'], data['true_classes']) if pred == true)
        
        analysis['summary']['label_accuracy_with_prior'] = correct_labels / len(data['true_labels'])
        analysis['summary']['class_accuracy_with_prior'] = correct_classes / len(data['true_classes'])
    
    return analysis

# Example usage:
# analysis = load_and_analyze_predictions("path/to/comprehensive_predictions.npy")
# print("Summary:", analysis['summary'])
# print("Label accuracy without prior:", analysis['summary'].get('label_accuracy_no_prior', 'N/A'))
# print("Class accuracy without prior:", analysis['summary'].get('class_accuracy_no_prior', 'N/A'))