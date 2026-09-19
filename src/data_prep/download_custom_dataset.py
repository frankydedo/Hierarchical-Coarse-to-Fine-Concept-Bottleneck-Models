import os
import shutil
import random
# pyrefly: ignore [missing-import]
from icrawler.builtin import BingImageCrawler
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.taxonomy import cifar100_taxonomy

DATA_DIR = "data/custom_images"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
IMAGES_PER_CLASS = 60
TRAIN_SPLIT = 50

def main():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    for superclass, subclasses in cifar100_taxonomy.items():
        for subclass in subclasses:
            print(f"\n--- Downloading images for: {superclass} -> {subclass} ---")
            
            sc_train_dir = os.path.join(TRAIN_DIR, superclass, subclass)
            sc_test_dir = os.path.join(TEST_DIR, superclass, subclass)
            os.makedirs(sc_train_dir, exist_ok=True)
            os.makedirs(sc_test_dir, exist_ok=True)
            
            # Check if already downloaded
            if len(os.listdir(sc_train_dir)) >= TRAIN_SPLIT:
                print(f"Already downloaded {subclass}.")
                continue
                
            # Temporary download dir
            temp_dir = os.path.join(DATA_DIR, f"temp_{subclass.replace(' ', '_')}")
            os.makedirs(temp_dir, exist_ok=True)

            crawler = BingImageCrawler(storage={'root_dir': temp_dir})
            # Add filter for photos to ensure good quality
            crawler.crawl(keyword=f"{subclass} photo high resolution", max_num=IMAGES_PER_CLASS + 10, filters={'type': 'photo'})
            
            # Move and split
            downloaded_files = [f for f in os.listdir(temp_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            random.shuffle(downloaded_files)
            
            for i, f in enumerate(downloaded_files[:IMAGES_PER_CLASS]):
                src = os.path.join(temp_dir, f)
                if i < TRAIN_SPLIT:
                    dst = os.path.join(sc_train_dir, f"{i:03d}.jpg")
                else:
                    dst = os.path.join(sc_test_dir, f"{i - TRAIN_SPLIT:03d}.jpg")
                shutil.move(src, dst)
                
            # Cleanup temp dir
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Completed {len(downloaded_files[:IMAGES_PER_CLASS])} images for {subclass}.")

if __name__ == "__main__":
    main()
