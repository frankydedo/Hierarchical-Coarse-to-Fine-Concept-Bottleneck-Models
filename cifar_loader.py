import pickle
import numpy as np
import os

def load_cifar_meta(meta_path):
    """Loads the CIFAR-100 metadata file to get class names."""
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    
    # Decode bytes to strings
    coarse_names = [label.decode('utf-8') for label in meta[b'coarse_label_names']]
    fine_names = [label.decode('utf-8') for label in meta[b'fine_label_names']]
    
    return coarse_names, fine_names

def load_cifar_batch(file_path):
    """Loads a CIFAR-100 batch file (train or test) and reshapes the images."""
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    # Extract labels
    coarse_labels = np.array(batch[b'coarse_labels'])
    fine_labels = np.array(batch[b'fine_labels'])
    
    # Extract images and reshape from (N, 3072) to (N, 32, 32, 3)
    # The array is stored as RRR...GGG...BBB...
    # We reshape to (N, 3, 32, 32) and transpose to standard image shape (N, 32, 32, 3)
    data = batch[b'data']
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return images, coarse_labels, fine_labels

if __name__ == "__main__":
    # Example usage script
    archive_dir = "archive"
    meta_path = os.path.join(archive_dir, "meta")
    train_path = os.path.join(archive_dir, "train")
    
    if not os.path.exists(meta_path):
        print(f"Error: Could not find {meta_path}.")
        exit(1)
        
    print("--- Loading CIFAR-100 Metadata ---")
    coarse_names, fine_names = load_cifar_meta(meta_path)
    print(f"Total Superclasses: {len(coarse_names)}")
    print(f"Total Subclasses: {len(fine_names)}")
    print(f"First 5 Superclasses: {coarse_names[:5]}")
    
    print("\n--- Loading CIFAR-100 Training Data ---")
    train_images, train_coarse, train_fine = load_cifar_batch(train_path)
    print(f"Training images shape: {train_images.shape} (Count, Height, Width, Channels)")
    
    print("\n--- Checking Image 0 ---")
    sample_idx = 0
    coarse_id = train_coarse[sample_idx]
    fine_id = train_fine[sample_idx]
    
    print(f"Image {sample_idx} is a '{fine_names[fine_id]}' which belongs to the '{coarse_names[coarse_id]}' superclass.")
