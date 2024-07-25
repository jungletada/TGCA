import os
import random
import shutil

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_images(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

def copy_images(images, src_folder, dest_folder):
    for image in images:
        shutil.copy(os.path.join(src_folder, image), os.path.join(dest_folder, image))

def main():
    folder1 = 'datasets/VOCdevkit/VOC2012/SegmentationClassAug'
    folder2 = 'results_voc/mctgformer_448/pseudo_mask'
    folder3 = 'results_voc/mctgformer_448/pseudo_mask_448'
    
    # Create folder3 if it does not exist
    create_folder(folder3)
    
    # Get list of images in both folders
    images_folder1 = get_images(folder1)
    images_folder2 = get_images(folder2)
    
    print("Number of images: {} and {}".format(len(images_folder1), len(images_folder2)))
    # Check for common images
    common_images = set(images_folder1).intersection(images_folder2)
    
    if not common_images:
        print("No common images found in folder1 and folder2")
        return
    
    # Calculate number of images to select
    alpha = 0.13
    num_images_folder1 = int(len(common_images) * alpha)
    num_images_folder2 = len(common_images) - num_images_folder1
    
    # Randomly select images
    selected_images_folder1 = random.sample(sorted(common_images), num_images_folder1)
    remaining_images = common_images - set(selected_images_folder1)
    selected_images_folder2 = random.sample(sorted(remaining_images), num_images_folder2)
    
    # Copy selected images to folder3
    copy_images(selected_images_folder1, folder1, folder3)
    copy_images(selected_images_folder2, folder2, folder3)
    
    print(f"Copied {len(selected_images_folder1)} images from folder1 and {len(selected_images_folder2)} images from folder2 to folder3.")

if __name__ == "__main__":
    main()
