import os
from PIL import Image
from cifar_loader import load_cifar_meta, load_cifar_batch

def extract_images(output_dir, data_path, meta_path, prefix="train"):
    coarse_names, fine_names = load_cifar_meta(meta_path)
    images, coarse_labels, fine_labels = load_cifar_batch(data_path)
    
    for i in range(len(images)):
        coarse_name = coarse_names[coarse_labels[i]]
        fine_name = fine_names[fine_labels[i]]
        
        # Create directory: output_dir/train/super_class/sub_class/
        save_dir = os.path.join(output_dir, prefix, coarse_name, fine_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # CIFAR arrays are natively ready for PIL once reshaped
        img = Image.fromarray(images[i])
        img.save(os.path.join(save_dir, f"{i}.png"))
        
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1} images from {prefix}...")

if __name__ == "__main__":
    archive_dir = "archive"
    meta_path = os.path.join(archive_dir, "meta")
    train_path = os.path.join(archive_dir, "train")
    test_path = os.path.join(archive_dir, "test")
    output_dir = "cifar_images"
    
    print("Extracting train images (this might take a minute)...")
    extract_images(output_dir, train_path, meta_path, "train")
    
    print("Extracting test images...")
    extract_images(output_dir, test_path, meta_path, "test")
    
    print("Done! All images are neatly organized in the 'cifar_images' directory.")
