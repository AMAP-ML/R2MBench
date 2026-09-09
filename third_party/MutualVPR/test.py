import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple
from argparse import Namespace
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, Dataset
import myparser
import torch.nn as nn
from cosplace_model import cosplace_network
from datasets.pittsburgh import get_whole_test_set,get_250k_test_set
from datasets.tokyo247 import PlaceDataset
from sklearn.decomposition import PCA
from datasets.test_dataset import TestDataset
# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]
args = myparser.parse_arguments()
print(f"There are {torch.cuda.device_count()} GPUs")



def test(args: Namespace, eval_ds: Dataset, model: torch.nn.Module,
        ) -> Tuple[np.ndarray, str]:
    """Compute descriptors of the given dataset and compute the recalls."""
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")
        for images, indices in tqdm(database_dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
        
        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        for images, indices in tqdm(queries_dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
    
    # pca = PCA(n_components=512)
    # all_descriptors = pca.fit_transform(all_descriptors)
    # all_descriptors = np.ascontiguousarray(all_descriptors)

    queries_descriptors = all_descriptors[eval_ds.database_num:]
    database_descriptors = all_descriptors[:eval_ds.database_num]

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors
    
    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))
    
    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    
    
    X = np.zeros((queries_descriptors.shape[0],10))

    recalls = np.zeros(len(RECALL_VALUES))
    recalls1 = np.zeros((queries_descriptors.shape[0],1))
    for query_index, preds in enumerate(predictions):
        if np.any(X[query_index]):
            recalls1[query_index] = 1
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])
    
    del faiss_index, queries_descriptors, predictions, positives_per_query, database_dataloader
    return recalls, recalls_str


if __name__ == "__main__":
    # Load the dataset and model
    test_ds = TestDataset('/your/path/processed/test', queries_folder="queries_occlusion",
                      positive_dist_threshold=args.positive_dist_threshold,
                      image_size=args.image_size, resize_test_imgs=args.resize_test_imgs)
    model = cosplace_network.MutualVPR(pretrained_foundation = True, foundation_model_path = '/your/path/dinov2_vitb14_pretrain.pth',output_dim=args.fc_output_dim)
    model = model.to(args.device)

    best_model_state_dict = torch.load('')
    model.load_state_dict(best_model_state_dict,strict=False)
    model = nn.DataParallel(model)
    # Compute recallsn
    recalls, recalls_str = test(args, test_ds, model)
    logging.info(f"Recalls: {recalls_str}")
    print(recalls_str)
    # Save the recalls
    with open(args.output_folder + "/recalls.txt", "w") as f:
        f.write(f"Recalls: {recalls_str}")
    logging.info(f"Recalls saved in {args.output_folder}/recalls.txt")






