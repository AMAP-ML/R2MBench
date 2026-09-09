
import os
import logging
from glob import glob
from PIL import ImageFile
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

PANO_WIDTH = int(512*6.5)
def read_images_paths(dataset_folder, get_abs_path=False):
    """Find images within 'dataset_folder' and return their relative paths as a list.
    If there is a file 'dataset_folder'_images_paths.txt, read paths from such file.
    Otherwise, use glob(). Keeping the paths in the file speeds up computation,
    because using glob over large folders can be slow.
    
    Parameters
    ----------
    dataset_folder : str, folder containing JPEG images
    get_abs_path : bool, if True return absolute paths, otherwise remove
        dataset_folder from each path
    
    Returns
    -------
    images_paths : list[str], paths of JPEG images within dataset_folder
    """
    
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")
    
    file_with_paths = dataset_folder + "/train_images_paths.txt"
    if os.path.exists(file_with_paths):
        logging.debug(f"Reading paths of images within {dataset_folder} from {file_with_paths}")
        with open(file_with_paths, "r") as file:
            images_paths = file.read().splitlines()
        images_paths = [os.path.join(dataset_folder, 'train',path) for path in images_paths]
        # Sanity check that paths within the file exist
        if not os.path.exists(images_paths[0]):
            raise FileNotFoundError(f"Image with path {images_paths[0]} "
                                    f"does not exist within {dataset_folder}. It is likely "
                                    f"that the content of {file_with_paths} is wrong.")
    else:
        logging.debug(f"Searching images in {dataset_folder} with glob()")
        images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
        if len(images_paths) == 0:
            raise FileNotFoundError(f"Directory {dataset_folder} does not contain any JPEG images")
    
    # if not get_abs_path:  # Remove dataset_folder from the path
    #     images_paths = [p[len(dataset_folder) + 1:] for p in images_paths]
    
    return images_paths



def read_pano_images_paths(dataset_folder, get_abs_path=False):
    """Find images within 'dataset_folder' and return their relative paths as a list.
    If there is a file 'dataset_folder'_images_paths.txt, read paths from such file.
    Otherwise, use glob(). Keeping the paths in the file speeds up computation,
    because using glob over large folders can be slow.
    
    Parameters
    ----------
    dataset_folder : str, folder containing JPEG images
    get_abs_path : bool, if True return absolute paths, otherwise remove
        dataset_folder from each path
    
    Returns
    -------
    images_paths : list[str], paths of JPEG images within dataset_folder
    """
    
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")
    
    file_with_paths = dataset_folder + "/panoramas_images_paths.txt"
    if os.path.exists(file_with_paths):
        logging.debug(f"Reading paths of images within {dataset_folder} from {file_with_paths}")
        with open(file_with_paths, "r") as file:
            images_paths = file.read().splitlines()
        if "test" in dataset_folder:
            images_paths = [os.path.join(dataset_folder, path) for path in images_paths]
        else:
            images_paths = [os.path.join(dataset_folder, 'train',path) for path in images_paths]
        # Sanity check that paths within the file exist
        if not os.path.exists(images_paths[0]):
            raise FileNotFoundError(f"Image with path {images_paths[0]} "
                                    f"does not exist within {dataset_folder}. It is likely "
                                    f"that the content of {file_with_paths} is wrong.")
    else:
        logging.debug(f"Searching images in {dataset_folder} with glob()")
        images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
        if len(images_paths) == 0:
            raise FileNotFoundError(f"Directory {dataset_folder} does not contain any JPEG images")
    
    # if not get_abs_path:  # Remove dataset_folder from the path
    #     images_paths = [p[len(dataset_folder) + 1:] for p in images_paths]
    
    return images_paths

def get_crop_image(pano_path, degree_offset, crop_size=512):
    pano_pil = Image.open(pano_path)
    start= int(degree_offset / 360 * PANO_WIDTH)
    images = []
    for i in range(6):
        offset = start + i * crop_size
        if offset +  crop_size <= PANO_WIDTH:
            pil_crop = pano_pil.crop((offset, 0, offset + crop_size, 512))
        else:
            crop1 = pano_pil.crop((offset, 0, PANO_WIDTH, 512))
            crop2 = pano_pil.crop((0, 0, crop_size - (PANO_WIDTH - offset), 512))
            pil_crop = Image.new('RGB', (crop_size, 512))
            pil_crop.paste(crop1, (0, 0))
            pil_crop.paste(crop2, (crop1.size[0], 0))
        images.append(pil_crop)
    return images

if __name__ == "__main__":
    def crop_pano_images():
        with open('/your/path/dataset_images_paths.txt','w') as f:
            images_paths = read_pano_images_paths('/your/path/panoramas')
            offset = 30
            for image_path in tqdm(images_paths, ncols=100):
                images = get_crop_image(image_path, offset,554)
                for i,image in enumerate(images):
                    image_path = image_path.replace('panoramas','crops_30')
                    path,name = os.path.split(image_path)
                    name = name.split('@')
                    name[10] = str(30+i*60)
                    name = '@'.join(name)
                    new_image_path = os.path.join(path,name)
                    if not os.path.exists(os.path.dirname(new_image_path)):
                        os.makedirs(os.path.dirname(new_image_path))
                    image.save(new_image_path)
                    f.write(new_image_path + '\n')


    with open('/your/path/crops_30_1/train_images_paths.txt','w') as f:
        for i in range(70,82):
            images_paths = sorted(glob(f"/your/path/crops_30_1/train/37.{str(i)}/*.jpg", recursive=True))
            for image_path in images_paths:
                image_path = image_path.replace('/your/path/crops_30_1s/train','')
                f.write(image_path + '\n')