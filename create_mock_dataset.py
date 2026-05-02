import os
from PIL import Image
import random

def create_dummy_images(base_dir, num_images=10):
    classes = ['O', 'R']
    splits = ['train', 'val']
    
    for split in splits:
        for cls in classes:
            dir_path = os.path.join(base_dir, split, cls)
            os.makedirs(dir_path, exist_ok=True)
            
            # create dummy images
            for i in range(num_images):
                img = Image.new('RGB', (224, 224), color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                img.save(os.path.join(dir_path, f'dummy_{i}.jpg'))
                
    print(f"Created dummy dataset in {base_dir}")

if __name__ == "__main__":
    create_dummy_images("data")
