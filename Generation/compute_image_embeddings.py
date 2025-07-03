#!/usr/bin/env python3
"""
Script to compute CLIP embeddings for images in a directory.

This script:
1. Reads all images from a specified directory
2. Computes CLIP embeddings using the same model as in the training scripts
3. Saves embeddings as numpy arrays with corresponding labels
4. Creates metadata JSON file with model information

Usage:
    python compute_image_embeddings.py --input_dir /path/to/images --output_dir /path/to/save/embeddings
"""

import os
import argparse
import torch
import numpy as np
import json
import datetime
from PIL import Image
import open_clip
from pathlib import Path
import time

def compute_image_embeddings(input_dir, output_dir, device="cuda:0"):
    """
    Compute CLIP embeddings for all images in the input directory.
    
    Args:
        input_dir: Directory containing images
        output_dir: Directory to save embeddings and metadata
        device: Device to use for computation
    
    Returns:
        Dictionary with metadata about the computation
    """
    print(f"Computing CLIP embeddings for images in: {input_dir}")
    print(f"Saving results to: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CLIP model (same as in training scripts)
    print("Loading CLIP model...")
    model, preprocess, _ = open_clip.create_model_and_transforms(
        'ViT-H-14',
        pretrained="./variables/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
        precision='fp32', 
        device=device
    )
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for file in os.listdir(input_dir):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(file)
    
    if not image_files:
        raise ValueError(f"No image files found in {input_dir}")
    
    print(f"Found {len(image_files)} image files")
    
    # Sort files for deterministic ordering
    image_files.sort()
    
    # Compute embeddings
    embeddings = []
    labels = []
    successful_files = []
    failed_files = []
    
    print("Computing embeddings...")
    start_time = time.time()
    
    for i, filename in enumerate(image_files):
        file_path = os.path.join(input_dir, filename)
        label = Path(filename).stem  # Remove extension for label
        
        try:
            # Load and preprocess image
            image = Image.open(file_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            # Compute embedding
            with torch.no_grad():
                image_feature = model.encode_image(image_input)
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
            
            # Store results
            embeddings.append(image_feature.squeeze(0).cpu().numpy())
            labels.append(label)
            successful_files.append(filename)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed_files.append(filename)
    
    computation_time = time.time() - start_time
    
    # Convert to numpy arrays
    embeddings_array = np.array(embeddings)
    labels_array = np.array(labels)
    
    print(f"Successfully processed {len(successful_files)}/{len(image_files)} images")
    print(f"Computation time: {computation_time:.2f} seconds")
    
    # Save embeddings and labels
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    labels_path = os.path.join(output_dir, "labels.npy")
    
    np.save(embeddings_path, embeddings_array)
    np.save(labels_path, labels_array)
    
    print(f"Saved embeddings to: {embeddings_path}")
    print(f"Saved labels to: {labels_path}")
    
    # Create metadata
    metadata = {
        "computation_info": {
            "timestamp": datetime.datetime.now().isoformat(),
            "computation_time_seconds": computation_time,
            "total_images_found": len(image_files),
            "successful_embeddings": len(successful_files),
            "failed_images": len(failed_files),
            "success_rate": len(successful_files) / len(image_files)
        },
        "model_info": {
            "model_name": "ViT-H-14",
            "pretrained_file": "./variables/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
            "precision": "fp32",
            "embedding_dimension": embeddings_array.shape[1],
            "normalized": True
        },
        "input_info": {
            "input_directory": os.path.abspath(input_dir),
            "supported_extensions": list(image_extensions),
            "failed_files": failed_files
        },
        "output_info": {
            "output_directory": os.path.abspath(output_dir),
            "embeddings_file": "embeddings.npy",
            "labels_file": "labels.npy",
            "metadata_file": "metadata.json"
        }
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to: {metadata_path}")
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Compute CLIP embeddings for images in a directory")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save embeddings and metadata")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use for computation (cuda:0, cpu, etc.)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input path is not a directory: {args.input_dir}")
    
    # Check if CLIP model file exists
    clip_model_path = "./variables/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
    if not os.path.exists(clip_model_path):
        raise FileNotFoundError(f"CLIP model not found at: {clip_model_path}")
    
    # Set device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Compute embeddings
    try:
        metadata = compute_image_embeddings(args.input_dir, args.output_dir, device)
        
        print("\n" + "="*60)
        print("COMPUTATION SUMMARY")
        print("="*60)
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Total images found: {metadata['computation_info']['total_images_found']}")
        print(f"Successful embeddings: {metadata['computation_info']['successful_embeddings']}")
        print(f"Failed images: {metadata['computation_info']['failed_images']}")
        print(f"Success rate: {metadata['computation_info']['success_rate']:.2%}")
        print(f"Computation time: {metadata['computation_info']['computation_time_seconds']:.2f} seconds")
        print(f"Embedding dimension: {metadata['model_info']['embedding_dimension']}")
        print("="*60)
        
    except Exception as e:
        print(f"Error during computation: {e}")
        raise

if __name__ == "__main__":
    main() 