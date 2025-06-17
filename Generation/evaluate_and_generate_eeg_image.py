import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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

# Import ATMS model components from training script
from train_ATMS_eeg_image import ATMS, EEGImageDataset, precompute_image_embeddings

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

def compute_eeg_embeddings(eeg_model, dataloader, device):
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

def evaluate_model(eeg_model, eeg_embeddings, labels, classes, precomputed_embeddings, unique_labels, device, k, use_prior=False, pipe=None, guidance_scale=5.0):
    """
    Evaluate model performance for image retrieval and classification.
    
    Args:
        eeg_model: The EEG model
        eeg_embeddings: EEG embeddings tensor
        labels: List of image labels
        classes: List of class labels
        precomputed_embeddings: Dictionary of precomputed image embeddings
        unique_labels: List of unique labels corresponding to embeddings
        device: Device to run on
        k: Number of classes to consider for evaluation
        use_prior: Whether to use diffusion prior
        pipe: Diffusion prior pipeline (required if use_prior=True)
        guidance_scale: Guidance scale for diffusion prior generation
    
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
    
    # Create embeddings tensor for all valid labels
    img_embeddings = torch.stack([precomputed_embeddings[label] for label in unique_labels]).to(device).float()
    all_label_indices = set(range(len(unique_labels)))
    
    with torch.no_grad():
        for idx, (eeg_embed, label, cls) in enumerate(zip(eeg_embeddings, labels, classes)):
            if label not in label_to_idx:
                continue
                
            true_label_idx = label_to_idx[label]
            
            # Generate embedding using diffusion prior if specified
            if use_prior and pipe is not None:
                h = pipe.generate(c_embeds=eeg_embed.unsqueeze(0), num_inference_steps=50, guidance_scale=guidance_scale).squeeze(0)
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
    Main function for EEG-based image generation and retrieval evaluation using trained ATMS model.
    
    Folder Structure Created:
    experiment_{experiment_id}/
    ├── models/
    │   └── ATMS/
    │       └── {timestamp}/
    │           ├── best_model.pth
    │           └── final_model.pth
    ├── evaluation_results/
    │   └── {timestamp}/
    │       ├── comprehensive_retrieval_results.csv
    │       ├── detailed_results.txt
    │       └── comprehensive_predictions.npy
    ├── generated_images/
    │   ├── {timestamp}_with_prior/
    │   │   └── label_{label}_gen_{j}.png
    │   └── {timestamp}_without_prior/
    │       └── label_{label}_gen_{j}.png
    ├── regenerated_stimuli/
    │   └── {timestamp}/
    │       └── {class}_gen_{j}.png
    └── diffusion_prior/
        └── diffusion_prior.pt
    """
    parser = argparse.ArgumentParser(description="EEG Image Generation and Retrieval Evaluation with ATMS")
    parser.add_argument("--task", type=str, default="evaluate", choices=["generate", "evaluate", "regenerate_stimuli"], help="Task to perform")
    parser.add_argument("--experiment_id", type=str, default="gtec_250527_data_250527_train_from_scratch", help="Experiment ID")
    parser.add_argument("--model_path", type=str, help="Path to trained ATMS model (if not provided, will use final_model.pth from latest training)")
    parser.add_argument("--evaluation_with_prior", action="store_true", help="Use diffusion prior for evaluation")
    parser.add_argument("--generate_without_prior", action="store_true", help="Generate images without diffusion prior")
    parser.add_argument("--generation_repeats", type=int, default=3, help="How many times to generate images for each embedding")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale for diffusion prior generation")
    args = parser.parse_args()

    experiment_folder = f"./experiments/experiment_{args.experiment_id}"
    data_path = os.path.join(experiment_folder, 'whitened_eeg_data.npy')
    stimuli_folder = os.path.join(experiment_folder, 'image_pool')
    diffusion_prior_folder = os.path.join(experiment_folder, 'diffusion_prior')
    
    # Create output directories
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    evaluation_results_folder = os.path.join(experiment_folder, 'evaluation_results', current_time)
    generated_images_folder = os.path.join(experiment_folder, 'generated_images')
    os.makedirs(evaluation_results_folder, exist_ok=True)
    os.makedirs(generated_images_folder, exist_ok=True)
    os.makedirs(diffusion_prior_folder, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print('Loading EEG dataset...')
    train_dataset = EEGImageDataset(data_path, train=True)
    test_dataset = EEGImageDataset(data_path, train=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    print('Loading trained ATMS model...')
    eeg_model = ATMS(num_channels=train_dataset.X.shape[1], sequence_length=250)
    
    # Load model weights
    if args.model_path:
        model_path = args.model_path
    else:
        # Find the latest training folder and use final_model.pth
        models_dir = os.path.join(experiment_folder, 'models', 'ATMS')
        if os.path.exists(models_dir):
            training_folders = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]
            if training_folders:
                latest_folder = sorted(training_folders)[-1]
                model_path = os.path.join(models_dir, latest_folder, 'final_model.pth')
            else:
                raise FileNotFoundError("No trained models found. Please train a model first or specify --model_path")
        else:
            raise FileNotFoundError("No models directory found. Please train a model first or specify --model_path")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from: {model_path}")
    eeg_model.load_state_dict(torch.load(model_path, map_location=device))
    eeg_model = eeg_model.to(device)

    print('Computing EEG embeddings...')
    emb_eeg_train, labels_train, classes_train = compute_eeg_embeddings(eeg_model, train_loader, device)
    emb_eeg_test, labels_test, classes_test = compute_eeg_embeddings(eeg_model, test_loader, device)

    # Precompute image embeddings with error handling
    print('Precomputing image embeddings...')
    all_labels = list(labels_train) + list(labels_test)
    precomputed_embeddings, valid_labels = precompute_image_embeddings(stimuli_folder, all_labels, device)
    
    # Filter training data to match valid labels
    valid_train_indices = [i for i, label in enumerate(labels_train) if label in valid_labels]
    if len(valid_train_indices) != len(labels_train):
        emb_eeg_train = emb_eeg_train[valid_train_indices]
        labels_train = [labels_train[i] for i in valid_train_indices]
        classes_train = [classes_train[i] for i in valid_train_indices]

    gen_repeats = args.generation_repeats
    
    if args.task == "regenerate_stimuli":
        print('Regenerating images for each unique test image...')

        # Create timestamped subfolder for regenerated stimuli
        regenerated_stimuli_folder = os.path.join(experiment_folder, 'regenerated_stimuli', current_time)
        os.makedirs(regenerated_stimuli_folder, exist_ok=True)

        # Regenerate images for each class
        generator = Generator4Embeds(num_inference_steps=50, device=device)
        for idx, label in enumerate(valid_labels):
            embed = precomputed_embeddings[label]
            for j in range(gen_repeats):
                image = generator.generate(embed.unsqueeze(0).to(dtype=torch.float16))
                image_path = os.path.join(regenerated_stimuli_folder, f"{label}_gen_{j}.png")
                image.save(image_path)
                print(f"Regenerated image saved to {image_path}")
        
        print(f"All regenerated stimuli saved to {regenerated_stimuli_folder}")
        return

    print('Creating embedding dataset...')
    # Get embeddings for training labels
    train_img_embeddings = torch.stack([precomputed_embeddings[label] for label in labels_train]).to(device)
    embedding_dataset = EmbeddingDataset(c_embeddings=emb_eeg_train, h_embeddings=train_img_embeddings)
    embedding_loader = DataLoader(embedding_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)

    # Setup diffusion prior only if needed
    pipe = None
    if (args.task == "generate" and not args.generate_without_prior) or (args.task == "evaluate" and args.evaluation_with_prior):
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
        method_suffix = "_without_prior" if args.generate_without_prior else "_with_prior"
        generated_images_timestamped_folder = os.path.join(generated_images_folder, current_time + method_suffix)
        os.makedirs(generated_images_timestamped_folder, exist_ok=True)

        for k in range(len(emb_eeg_test)):
            eeg_embeds = emb_eeg_test[k:k+1]
            label = labels_test[k]
            
            if args.generate_without_prior:
                # Generate directly from EEG embeddings without diffusion prior
                print("WARNING: Generating without diffusion prior may produce poor results due to domain mismatch between EEG and CLIP embedding spaces.")
                h = eeg_embeds.to(dtype=torch.float16)
            else:
                # Use diffusion prior to generate intermediate embeddings
                h = pipe.generate(c_embeds=eeg_embeds, num_inference_steps=50, guidance_scale=args.guidance_scale)
            
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
        print("="*60)
        
    elif args.task == "evaluate":
        # Filter test data to match valid labels
        valid_test_indices = [i for i, label in enumerate(labels_test) if label in valid_labels]
        if len(valid_test_indices) != len(labels_test):
            emb_eeg_test = emb_eeg_test[valid_test_indices]
            labels_test = [labels_test[i] for i in valid_test_indices]
            classes_test = [classes_test[i] for i in valid_test_indices]

        print('Performing comprehensive retrieval and classification evaluation...')
        
        # Evaluation configurations
        k_values = [2, 4, 10, 50, 100, len(valid_labels)]
        
        # Results storage
        results = {}
        all_predictions = {}
        
        # Evaluate without diffusion prior
        print('Evaluating without diffusion prior...')
        for k in k_values:
            if k > len(valid_labels):
                k = len(valid_labels)
            
            acc, top5_acc, class_acc, predictions_dict = evaluate_model(
                eeg_model, emb_eeg_test, labels_test, classes_test, 
                precomputed_embeddings, valid_labels, device, k, 
                use_prior=False, pipe=None, guidance_scale=args.guidance_scale
            )
            
            results[f'k={k}_retrieval_acc'] = acc
            results[f'k={k}_class_acc'] = class_acc
            if k >= 5:
                results[f'k={k}_top5_acc'] = top5_acc
            
            # Store predictions for the full dataset evaluation (largest k)
            if k == len(valid_labels):
                all_predictions['no_prior'] = predictions_dict
            
            print(f"k={k}: Retrieval Acc={acc:.4f}, Class Acc={class_acc:.4f}, Top5 Acc={top5_acc:.4f}")
        
        # Evaluate with diffusion prior if enabled
        if args.evaluation_with_prior:
            print('Evaluating with diffusion prior...')
            for k in k_values:
                if k > len(valid_labels):
                    k = len(valid_labels)
                
                acc, top5_acc, class_acc, predictions_dict = evaluate_model(
                    eeg_model, emb_eeg_test, labels_test, classes_test, 
                    precomputed_embeddings, valid_labels, device, k, 
                    use_prior=True, pipe=pipe, guidance_scale=args.guidance_scale
                )
                
                results[f'k={k}_retrieval_acc_with_prior'] = acc
                results[f'k={k}_class_acc_with_prior'] = class_acc
                if k >= 5:
                    results[f'k={k}_top5_acc_with_prior'] = top5_acc
                
                # Store predictions for the full dataset evaluation (largest k)
                if k == len(valid_labels):
                    all_predictions['with_prior'] = predictions_dict
                
                print(f"k={k} (with prior): Retrieval Acc={acc:.4f}, Class Acc={class_acc:.4f}, Top5 Acc={top5_acc:.4f}")
        
        # Save comprehensive results
        print('Saving comprehensive results...')
        
        # Save to CSV
        csv_path = os.path.join(evaluation_results_folder, "comprehensive_retrieval_results.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            for metric, value in results.items():
                writer.writerow([metric, value])
        
        # Save detailed results to text file
        txt_path = os.path.join(evaluation_results_folder, "detailed_results.txt")
        with open(txt_path, "w") as f:
            f.write("Comprehensive Retrieval and Classification Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Image Retrieval Accuracies (without prior):\n")
            for k in k_values:
                if k > len(valid_labels):
                    k = len(valid_labels)
                if f'k={k}_retrieval_acc' in results:
                    f.write(f"  k={k}: {results[f'k={k}_retrieval_acc']:.4f}\n")
            
            f.write("\nClass Classification Accuracies (without prior):\n")
            for k in k_values:
                if k > len(valid_labels):
                    k = len(valid_labels)
                if f'k={k}_class_acc' in results:
                    f.write(f"  k={k}: {results[f'k={k}_class_acc']:.4f}\n")
            
            f.write("\nTop-5 Accuracies (without prior):\n")
            for k in k_values:
                if k > len(valid_labels):
                    k = len(valid_labels)
                if f'k={k}_top5_acc' in results:
                    f.write(f"  k={k}: {results[f'k={k}_top5_acc']:.4f}\n")
            
            if args.evaluation_with_prior:
                f.write("\nImage Retrieval Accuracies (with prior):\n")
                for k in k_values:
                    if k > len(valid_labels):
                        k = len(valid_labels)
                    if f'k={k}_retrieval_acc_with_prior' in results:
                        f.write(f"  k={k}: {results[f'k={k}_retrieval_acc_with_prior']:.4f}\n")
                
                f.write("\nClass Classification Accuracies (with prior):\n")
                for k in k_values:
                    if k > len(valid_labels):
                        k = len(valid_labels)
                    if f'k={k}_class_acc_with_prior' in results:
                        f.write(f"  k={k}: {results[f'k={k}_class_acc_with_prior']:.4f}\n")
                
                f.write("\nTop-5 Accuracies (with prior):\n")
                for k in k_values:
                    if k > len(valid_labels):
                        k = len(valid_labels)
                    if f'k={k}_top5_acc_with_prior' in results:
                        f.write(f"  k={k}: {results[f'k={k}_top5_acc_with_prior']:.4f}\n")
            
            f.write(f"\nTotal test samples: {len(labels_test)}\n")
            f.write(f"Unique labels: {len(valid_labels)}\n")
            f.write(f"Unique classes: {len(set(classes_test))}\n")
        
        print(f"Results saved to {csv_path} and {txt_path}")
        
        # Save comprehensive predictions for detailed analysis
        print('Saving comprehensive predictions for detailed analysis...')
        
        # Prepare comprehensive predictions data
        predictions_data = {
            "unique_test_labels": valid_labels,
            "total_test_samples": len(labels_test),
            "unique_classes": list(set(classes_test)),
            "model_path": model_path
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
        predictions_path = os.path.join(evaluation_results_folder, "comprehensive_predictions.npy")
        np.save(predictions_path, predictions_data)
        
        print(f"Comprehensive predictions saved to {predictions_path}")
        print("Comprehensive evaluation completed!")
        
        # Summary of evaluation
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Model used: {model_path}")
        print(f"Total test samples evaluated: {len(labels_test)}")
        print(f"Unique image labels: {len(valid_labels)}")
        print(f"Unique classes: {len(set(classes_test))}")
        print(f"K-values evaluated: {k_values}")
        print(f"Diffusion prior used: {'Yes' if args.evaluation_with_prior else 'No'}")
        print(f"Results saved to: {evaluation_results_folder}")
        print(f"CSV results: {csv_path}")
        print(f"Detailed results: {txt_path}")
        print(f"Predictions data: {predictions_path}")
        print("="*60)

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

if __name__ == "__main__":
    main()

# Example usage:
# analysis = load_and_analyze_predictions("path/to/comprehensive_predictions.npy")
# print("Summary:", analysis['summary'])
# print("Label accuracy without prior:", analysis['summary'].get('label_accuracy_no_prior', 'N/A'))
# print("Class accuracy without prior:", analysis['summary'].get('class_accuracy_no_prior', 'N/A')) 