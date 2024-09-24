import sys
import torch
from torchmeta.utils.data import BatchMetaDataLoader
from common.args import parse_args
from common.utils import get_optimizer, load_model
from data.dataset import get_meta_dataset
from models.model import get_model
from train.trainer import meta_trainer
from utils import Logger, set_random_seed, cycle
import argparse

def main(rank, P):
    P.rank = rank

    """ Set torch device """
    if torch.cuda.is_available():
        torch.cuda.set_device(P.rank)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    """ Fix randomness """
    set_random_seed(P.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    """ Define dataset and dataloader """
    kwargs = {'batch_size': P.batch_size, 'shuffle': True,
              'pin_memory': True, 'num_workers': 2}

    # Use the r1 and r2 arguments to create the meta dataset
    train_set, val_set = get_meta_dataset(P, dataset=P.dataset, r1=P.r1, r2=P.r2)

    train_loader = train_set
    test_loader = val_set

    """ Initialize model, optimizer, loss scaler (for amp), and scheduler """
    model = get_model(P, P.model).to(device)
    optimizer = get_optimizer(P, model)

    """ Define train and test type """
    from train import setup as train_setup
    from evals import setup as test_setup
    train_func, fname, today = train_setup(P.mode, P)
    test_func = test_setup(P.mode, P)

    """ Define logger """
    # use different log labels as needed
    #log_dir_name = f"{fname}_batch_{P.batch_size}_lr_{P.lr}_seed{P.seed}"
    #log_dir_name = f"{fname}_r1_{P.r1}_r2_{P.r2}_seed{P.seed}"
    log_dir_name = f"{fname}_trainingStep{P.outer_steps}_seed{P.seed}"
    logger = Logger(log_dir_name, ask=P.resume_path is None, today=today, rank=P.rank)
    logger.log(P)
    logger.log(model)

    """ Load model if necessary """
    if P.resume_path:
        try:
            print(f"Attempting to load the model from {P.resume_path}...")
            load_model(P, model, logger)
            print("Model successfully loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    else:
        print("No checkpoint path provided, starting training from scratch.")

    """ Train the model """
    meta_trainer(P, train_func, test_func, model, optimizer, train_loader, test_loader, logger, outer_steps=P.outer_steps)

    """ close tensorboard """
    logger.close_writer()

if __name__ == "__main__":
    """ Define arguments """
    parser = argparse.ArgumentParser(description="Main script for running STUNT")
    
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to the resume checkpoint")
    parser.add_argument("--num_ways", type=int, default=10, help="Number of ways for few-shot learning")
    parser.add_argument("--num_shots", type=int, default=1, help="Number of support shots (K) for training")
    parser.add_argument("--num_shots_test", type=int, default=15, help="Number of query shots for testing")
    parser.add_argument("--r1", type=float, default=0.3, help="Ratio 1 for dataset creation")
    parser.add_argument("--r2", type=float, default=0.5, help="Ratio 2 for dataset creation")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and testing")
    parser.add_argument("--test_batch_size", type=int, default=4, help="Batch size for the test loader")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--mode", type=str, default="protonet", help="Mode for the model")
    parser.add_argument("--model", type=str, default="mlp", help="Model name")
    parser.add_argument("--dataset", type=str, default="income", help="Dataset name")
    parser.add_argument("--outer_steps", type=int, default=10000, help="Number of outer steps for training")

    parser.add_argument("--regression", action="store_true", help="Use MSE loss for regression tasks")
    parser.add_argument("--baseline", action="store_true", help="Use baseline model without saving the data")
    parser.add_argument("--num_shots_global", type=int, default=0, help="Number of global shots (for distillation)")
    parser.add_argument("--suffix", type=str, default=None, help="Suffix for the log directory")
    parser.add_argument("--print_step", type=int, default=50, help="Steps interval to print training statistics")
    parser.add_argument("--eval_step", type=int, default=50, help="Steps interval to evaluate the model")  # Newly added
    parser.add_argument("--save_step", type=int, default=1250, help="Steps interval to save the model")  # Newly added
    parser.add_argument("--max_test_task", type=int, default=1000, help="Max number of tasks for inference")  # Newly added

    P = parser.parse_args()

    P.world_size = torch.cuda.device_count()
    P.distributed = P.world_size > 1
    if P.distributed:
        print("currently, ddp is not supported, should consider transductive BN before using ddp",
              file=sys.stderr)
    else:
        main(0, P)
