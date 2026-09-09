
import torch
import shutil
import logging
from typing import Type, List
from argparse import Namespace
from cosface_loss import MarginCosineProduct
import random
import numpy as np
from sklearn import metrics

import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from tqdm import tqdm
import scipy.sparse as sp
def move_to_device(optimizer: Type[torch.optim.Optimizer], device: str):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def save_checkpoint(state: dict, is_best: bool, output_folder: str,
                    ckpt_filename: str = "last_checkpoint.pth"):
    # TODO it would be better to move weights to cpu before saving
    checkpoint_path = f"{output_folder}/{ckpt_filename}"
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state["model_state_dict"], f"{output_folder}/best_model.pth")


def resume_train(args: Namespace, output_folder: str, model: torch.nn.Module,
                 model_optimizer: Type[torch.optim.Optimizer], classifiers: List[MarginCosineProduct],
                 classifiers_optimizers: List[Type[torch.optim.Optimizer]]):
    """Load model, optimizer, and other training parameters"""
    logging.info(f"Loading checkpoint: {args.resume_train}")
    checkpoint = torch.load(args.resume_train)
    start_epoch_num = checkpoint["epoch_num"]
    
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)
    
    model = model.to(args.device)
    model_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    assert args.groups_num == len(classifiers) == len(classifiers_optimizers) == \
        len(checkpoint["classifiers_state_dict"]) == len(checkpoint["optimizers_state_dict"]), \
        (f"{args.groups_num}, {len(classifiers)}, {len(classifiers_optimizers)}, "
         f"{len(checkpoint['classifiers_state_dict'])}, {len(checkpoint['optimizers_state_dict'])}")
    
    for c, sd in zip(classifiers, checkpoint["classifiers_state_dict"]):
        # Move classifiers to GPU before loading their optimizers
        c = c.to(args.device)
        c.load_state_dict(sd)
    for c, sd in zip(classifiers_optimizers, checkpoint["optimizers_state_dict"]):
        c.load_state_dict(sd)
    for c in classifiers:
        # Move classifiers back to CPU to save some GPU memory
        c = c.cpu()
    
    best_val_recall1 = checkpoint["best_val_recall1"]
    
    # Copy best model to current output_folder
    shutil.copy(args.resume_train.replace("last_checkpoint.pth", "best_model.pth"), output_folder)
    
    return model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num



def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eva(y_true, y_pred, show_details=True):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if show_details:
        print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
              ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1

def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

