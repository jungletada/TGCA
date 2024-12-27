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


def copy_voc():
    path = 'results_voc/mcta'
    folder1 = 'data/VOCdevkit/VOC2012/SegmentationClassAug'
    folder2 = f'{path}/pseudo_mask_train'
    folder3 = f'{path}/pseudo_mask_448'
    
    # Create folder3 if it does not exist
    create_folder(folder3)
    
    # Get list of images in both folders
    images_folder1 = get_images(folder1)
    images_folder2 = get_images(folder2)
    
    print(f"Number of images: {len(images_folder1)} and {len(images_folder2)}")
    
    # Check for common images
    common_images = set(images_folder1).intersection(images_folder2)
    if not common_images:
        print("No common images found in folder1 and folder2")
        return
    
    # Calculate number of images to select
    alpha = 0.45
    num_images_folder1 = int(len(common_images) * alpha)
    num_images_folder2 = len(common_images) - num_images_folder1
    
    # Randomly select images
    selected_images_folder1 = random.sample(sorted(common_images), num_images_folder1)
    remaining_images = common_images - set(selected_images_folder1)
    selected_images_folder2 = random.sample(sorted(remaining_images), num_images_folder2)
    
    # Copy selected images to folder3
    copy_images(selected_images_folder1, folder1, folder3)
    copy_images(selected_images_folder2, folder2, folder3)
    
    print(f"Copied {len(selected_images_folder1)} images from folder1 \
          and {len(selected_images_folder2)} images from folder2 to folder3.")


def copy_coco():
    path = 'results_coco/mcta'
    gt_folder = 'data/MSCOCO/MaskSets/train2014'
    res_folder = f'{path}/pseudo_mask_train'
    combine_folder = f'{path}/pseudo_mask'
    list_path = 'data/MSCOCO/ImageLists/train_id.txt'
    
    file_names = None
    with open(list_path, 'r') as f:
        file_names = f.readlines()
    
    image_paths = [os.path.join(gt_folder, f.strip()+'.png') for f in file_names]
    # Create folder3 if it does not exist
    create_folder(combine_folder)
    
    # Get list of images in both folders
    images_gt_folder = get_images(gt_folder)
    images_res_folder = get_images(res_folder)
    
    print(f"Number of images: {len(images_gt_folder)} and {len(images_res_folder)}")
    
    # Check for common images
    common_images = set(images_gt_folder).intersection(images_res_folder)
    if not common_images:
        print("No common images found in folder1 and folder2")
        return
    
    # Calculate number of images to select
    alpha = 0.30
    num_images_gt_folder = int(len(common_images) * alpha)
    num_images_res_folder = len(common_images) - num_images_gt_folder
    
    # Randomly select images
    selected_images_folder1 = random.sample(sorted(common_images), num_images_gt_folder)
    remaining_images = common_images - set(selected_images_folder1)
    selected_images_folder2 = random.sample(sorted(remaining_images), num_images_res_folder)
    
    # Copy selected images to folder3
    copy_images(selected_images_folder1, gt_folder, combine_folder)
    copy_images(selected_images_folder2, res_folder, combine_folder)
    
    print(f"Copied {len(selected_images_folder1)} images from GT-folder \
          and {len(selected_images_folder2)} images from Result-folder to Combine-folder.")
    
    
if __name__ == "__main__":
    copy_coco()
