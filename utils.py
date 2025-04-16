import os
import random
import numpy as np
import torch
import os
import random
import numpy as np
import torch
from datetime import datetime
import shutil
import glob
from pathlib import Path
import time
import logging
from logging import Logger


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collate_func(x):
    smiles_embed, protein_embed, smiles_mask, protein_mask, labels = zip(*x)

    protein_embeddings = torch.cat(protein_embed, dim=0)
    smiles_embeddings = torch.cat(smiles_embed, dim=0)
    protein_masks = torch.cat(protein_mask, dim=0)
    smiles_masks = torch.cat(smiles_mask, dim=0)

    return [smiles_embeddings, protein_embeddings], [smiles_masks, protein_masks], torch.tensor(labels)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)
        

def get_checkpoint_name(epoch):
    """Generate checkpoint filename
    Args:
        epoch: Current epoch number
    Returns:
        str: Formatted checkpoint filename
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'checkpoint_latest_{timestamp}.pth'


def save_checkpoint(state, is_best, checkpoint_dir):
    """Save model checkpoint - keeps only latest and best
    Args:
        state: Dictionary containing state to save
        is_best: Boolean indicating if this is the best model so far
        checkpoint_dir: Directory path to save checkpoint
    """
    # Ensure directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save latest checkpoint
    latest_filename = get_checkpoint_name(state['epoch'])
    latest_filepath = os.path.join(checkpoint_dir, latest_filename)
    
    # Remove old latest checkpoint if exists
    old_latest = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_latest_*.pth'))
    for old_file in old_latest:
        os.remove(old_file)
        
    # Save new latest checkpoint
    torch.save(state, latest_filepath)
    print(f"Latest checkpoint saved: {latest_filepath}")
    
    # If this is the best model, save as best model
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth')
        shutil.copy(latest_filepath, best_filepath)
        print(f"Best model saved: {best_filepath}")


def load_latest_checkpoint(checkpoint_dir, start_epoch, model, optimizer_gen, optimizer_pred):
    """Load the latest checkpoint
    Args:
        checkpoint_dir: Checkpoint directory
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler to load state into
    Returns:
        epoch: Epoch to resume training from
        best_loss: Best loss achieved so far
    """
    # Try to load latest checkpoint first
    latest_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_latest_*.pth'))
    
    if latest_checkpoints:
        checkpoint_path = latest_checkpoints[0]
    else:
        # If no latest checkpoint, try to load best model
        best_model_path = os.path.join(checkpoint_dir, 'model_best.pth')
        if os.path.exists(best_model_path):
            checkpoint_path = best_model_path
        else:
            print("No checkpoints found")
            return 0, float('inf')
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])
    optimizer_pred.load_state_dict(checkpoint['optimizer_pred_state_dict'])
    
    return start_epoch, model, optimizer_gen, optimizer_pred


def construct_logger(logger_dir: Path) -> Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    os.makedirs(logger_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        f"{logger_dir}/log_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}.txt")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
