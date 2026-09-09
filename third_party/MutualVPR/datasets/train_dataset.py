
from email.mime import image
import os
from re import I
from types import new_class
import torch
import random
import logging
import numpy as np
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as T
from collections import defaultdict
import torch.utils.data as data
import datasets.dataset_utils as dataset_utils


ImageFile.LOAD_TRUNCATED_IMAGES = True
def find_indices(matrix, targets):
    res = list()
    for elem in targets:
        for i in range(len(matrix)):
            if elem >= len(matrix[i]):
                elem -= len(matrix[i])
            else:
                res.append((i,elem))
                break
    return res


    

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset_folder, M=10, alpha=30, N=5, C=6,
                 current_group=0, min_images_per_class=10,k=8):
        """
        Parameters (please check our paper for a clearer explanation of the parameters).
        ----------
        args : args for data augmentation
        dataset_folder : str, the path of the folder with the train images.
        M : int, the length of the side of each cell in meters.
        alpha : int, size of each class in degrees.
        N : int, distance (M-wise) between two classes of the same group.
        L : int, distance (alpha-wise) between two classes of the same group.
        current_group : int, which one of the groups to consider.
        min_images_per_class : int, minimum number of image in a class.
        """
        super().__init__()
        self.M = M
        self.alpha = alpha
        self.N = N
        self.C = C
        self.current_group = current_group
        self.dataset_folder = dataset_folder
        self.augmentation_device = args.augmentation_device
        self.k=k
        # dataset_name should be either "processed", "small" or "raw", if you're using SF-XL
        dataset_name = os.path.basename(dataset_folder)
        filename = f"cache/{dataset_name}_M{M}_N{N}_alpha{alpha}_C{C}_mipc{min_images_per_class}.torch"
        if not os.path.exists(filename):
            os.makedirs("cache", exist_ok=True)
            logging.info(f"Cached dataset {filename} does not exist, I'll create it now.")
            self.initialize(dataset_folder, M, N, alpha, C, min_images_per_class, filename)
        elif current_group == 0:
            logging.info(f"Using cached dataset {filename}")
        
        self.classes_per_group, self.images_per_class,self.utm_mean,self.utm_var = torch.load(filename)
        self.image_dict=dict()
        for class_id in self.classes_per_group[current_group]:
            if class_id[:2] not in self.image_dict:
                self.image_dict[class_id[:2]] = self.images_per_class[class_id].copy()
            else:
                self.image_dict[class_id[:2]].extend(self.images_per_class[class_id].copy())
        if current_group >= len(self.classes_per_group):
            raise ValueError(f"With this configuration there are only {len(self.classes_per_group)} " +
                             f"groups, therefore I can't create the {current_group}th group. " +
                             "You should reduce the number of groups by setting for example " +
                             f"'--groups_num {current_group}'")
        self.classes_ids = self.classes_per_group[current_group]
        self.origin_classes_ids = list(self.classes_ids)
        if self.augmentation_device == "cpu":
            self.transform = T.Compose([
                    T.ColorJitter(brightness=args.brightness,
                                  contrast=args.contrast,
                                  saturation=args.saturation,
                                  hue=args.hue),
                    T.RandomResizedCrop([args.image_size, args.image_size], scale=[1-args.random_resized_crop, 1], antialias=True),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        self.resize  = T.RandomResizedCrop(512, antialias=True)
        del self.classes_per_group
    @staticmethod
    def open_image(path):
        return Image.open(path).convert("RGB")
    
    def __getitem__(self, class_idx):
        if class_idx >= len(self.origin_classes_ids):
            class_idx = np.random.randint(0, len(self.origin_classes_ids) - 1)
        class_id = self.origin_classes_ids[class_idx]
        
        image_path =  random.choice(self.images_per_class[class_id])
        # image_path = os.path.join(self.dataset_folder, random.choice(self.images_per_class[class_id]))
                
        try:
            pil_image = TrainDataset.open_image(image_path)
        except Exception as e:
            logging.info(f"ERROR image {image_path} couldn't be opened, it might be corrupted.")
            raise e
        
        tensor_image = T.functional.to_tensor(pil_image)
        # assert tensor_image.shape == torch.Size([3, 512, 512]), \
        #     f"Image {image_path} should have shape [3, 512, 512] but has {tensor_image.shape}."
        if tensor_image.shape != torch.Size([3, 512, 512]):
            tensor_image = self.resize(tensor_image)
                
        return tensor_image, class_idx, class_id

    def get_random_class(self, min_samples=40):
        while True:
            sample = random.sample(list(self.image_dict.keys()), 1)
            image_list = self.image_dict[sample[0]]
            if len(image_list) >= min_samples:
                return image_list,sample[0]
            
    def get_class_images(self, class_id):
        return self.image_dict[class_id]
    
    def get_all_images(self):
        return [(key,images) for key,images in self.image_dict.items()]      
    
    def get_queries_class_images(self, queries_path):
        queries_class_images = defaultdict(dict)
        for query_path in queries_path:
            query = query_path.split("@")
            round_east = int(float(query[1]) // self.M * self.M)
            round_north = int(float(query[2]) // self.M * self.M)
            class_id = (round_east, round_north)
            if class_id in self.image_dict.keys():
                queries_class_images[class_id] = (self.image_dict[class_id])
                queries_class_images[class_id].append(query_path)
        return queries_class_images
        
    def get_images_num(self):
        """Return the number of images within this group."""
        return sum([len(self.images_per_class[c]) for c in self.classes_ids])
    
    def __len__(self):
        """Return the number of classes within this group."""
        return len(self.classes_ids)
    
    @staticmethod
    def initialize(dataset_folder, M, N, alpha, C, min_images_per_class, filename):
        logging.debug(f"Searching training images in {dataset_folder}")
        
        images_paths = dataset_utils.read_images_paths(dataset_folder, get_abs_path=True)

        # ***********************************************
        
        # if using cropping strategy
        # images_paths += dataset_utils.read_images_paths('/your/path/crops', get_abs_path=True)

        # ***********************************************
        logging.debug(f"Found {len(images_paths)} images")
        
        logging.debug("For each image, get its UTM east, UTM north and heading from its path")
        images_metadatas = [p.split("@") for p in images_paths]
        # field 1 is UTM east, field 2 is UTM north, field 9 is heading
        utmeast_utmnorth = [(m[1], m[2]) for m in images_metadatas]
        utmeast_utmnorth = np.array(utmeast_utmnorth).astype(np.float64)

        mean = np.mean(utmeast_utmnorth,0)
        x_mean = int(mean[0])
        y_mean = int(mean[1])
        var = np.var(utmeast_utmnorth,0)
        x_var = int(var[0])
        y_var = int(var[1])
        logging.debug("For each image, get class and group to which it belongs")
        class_id__group_id = [TrainDataset.get__class_id__group_id(*m, M, alpha, N, C)
                              for m in utmeast_utmnorth]
        
        logging.debug("Group together images belonging to the same class")
        images_per_class = defaultdict(list)
        for image_path, (class_id, group) in zip(images_paths, class_id__group_id):
            images_per_class[class_id].append(image_path)
            for i in range(1, C):
                images_per_class[class_id[:2]+(i,)]=list()
                class_id__group_id.append((class_id[:2]+(i,),group))
        for class_id in images_per_class:
            if class_id[2] != 0:
                images_per_class[class_id] = random.sample( images_per_class[class_id_t], num//C)
                indices = np.where(np.isin(images_per_class[class_id_t], images_per_class[class_id]))[0]
                images_per_class[class_id_t] = np.delete(np.array(images_per_class[class_id_t]), indices).tolist()
            else:
                class_id_t = class_id
                num = len(images_per_class[class_id])
        # Images_per_class is a dict where the key is class_id, and the value
        # is a list with the paths of images within that class.
        #images_per_class = {k: v for k, v in images_per_class.items() if len(v) >= min_images_per_class}
        
        logging.debug("Group together classes belonging to the same group")
        # Classes_per_group is a dict where the key is group_id, and the value
        # is a list with the class_ids belonging to that group.
        classes_per_group = defaultdict(set)
        for class_id, group_id in class_id__group_id:
            if class_id not in images_per_class:
                continue  # Skip classes with too few images
            classes_per_group[group_id].add(class_id)
        
        # Convert classes_per_group to a list of lists.
        # Each sublist represents the classes within a group.
        classes_per_group = [list(c) for c in classes_per_group.values()]
        classes_per_group = [sorted(c) for c in classes_per_group]
        
        
        torch.save((classes_per_group, images_per_class,(x_mean,y_mean),(x_var,y_var)), filename)
    
    @staticmethod
    def get__class_id__group_id(utm_east, utm_north, M, alpha, N, L):
        """Return class_id and group_id for a given point.
            The class_id is a triplet (tuple) of UTM_east, UTM_north and
            heading (e.g. (396520, 4983800,120)). 
            The group_id represents the group to which the class belongs
            (e.g. (0, 1, 0)), and it is between (0, 0, 0) and (N, N, L).
        """
        rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
        rounded_utm_north = int(utm_north // M * M)
        
        class_id = (rounded_utm_east, rounded_utm_north,0)
        
        group_id = (rounded_utm_east % (M * N) // M,
                    rounded_utm_north % (M * N) // M,
                    )
        return class_id, group_id
    
    
    def update_classes(self,labels,class_id):
        class_dict = dict()
        for i in range(self.C):
            class_dict[class_id + (i,)] = list()
        for label,path in zip(labels,self.image_dict[class_id]):
            class_dict[class_id + (label,)].append(path)
        self.images_per_class.update(class_dict)

    

class ImageKeyDataset(torch.utils.data.Dataset):
    def __init__(self, keys, group, transform):
        self.keys = keys
        self.group = group
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        images_path = self.group.image_dict[key]

        images_tensor = torch.empty(len(images_path), 3, 504, 504)
        for i, path in enumerate(images_path):
            pil_image = TrainDataset.open_image(path)
            images_tensor[i] = self.transform(pil_image)

        return key, images_tensor 
    
def custom_collate_fn(batch):
    keys, images = zip(*batch) 
    return list(keys), torch.stack(images) 
