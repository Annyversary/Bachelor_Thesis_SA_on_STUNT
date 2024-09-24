import time
from collections import OrderedDict

import torch
import torch.nn as nn

from common.utils import is_resume
from utils import MetricLogger, save_checkpoint, save_checkpoint_step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def meta_trainer(P, train_func, test_func, model, optimizer, train_loader, test_loader, logger, outer_steps=None):
    """
    Conducts the meta-training, including training, evaluation, and model checkpointing.

    Args:
        P: Parameters for the setup.
        train_func: Function to perform the training step.
        test_func: Function to perform the testing step.
        model: The model to be trained.
        optimizer: Optimizer for the model parameters.
        train_loader: DataLoader for the training data.
        test_loader: DataLoader for the test data.
        logger: Logger to log the training process.
        outer_steps: (Optional) Number of training steps to execute.
    """
    kwargs = {}  # Additional arguments for the training function
    kwargs_test = {}  # Additional arguments for the testing function

    # Initialize a MetricLogger to track metrics
    metric_logger = MetricLogger(delimiter="  ")

    """ Resume option """
    is_best, start_step, best, acc = is_resume(P, model, optimizer)

    """ Define loss function """
    criterion = nn.CrossEntropyLoss()

    """ Training start """
    logger.log_dirname("Start training")

    max_steps = outer_steps if outer_steps is not None else P.outer_steps

    for step in range(start_step, max_steps + 1):
        stime = time.time()
        train_batch = next(train_loader) 
        metric_logger.meters['data_time'].update(time.time() - stime)  

        train_func(P, step, model, criterion, optimizer, train_batch,
                   metric_logger=metric_logger, logger=logger, **kwargs)

        """ Evaluation & save the best model """
        if step % P.eval_step == 0:
            acc = test_func(P, model, test_loader, criterion, step, logger=logger, **kwargs_test)

            if best < acc:
                best = acc
                save_checkpoint(P, step, best, model.state_dict(),
                                optimizer.state_dict(), logger.logdir, is_best=True)

            logger.scalar_summary('eval/best_acc', best, step)
            logger.log('[EVAL] [Step %3d] [Acc %5.2f] [Best %5.2f]' % (step, acc, best))

        """ Save model every save_step steps """
        if step % P.save_step == 0:
            save_checkpoint_step(P, step, best, model.state_dict(),
                                 optimizer.state_dict(), logger.logdir)

    """ Save the last model """
    save_checkpoint(P, max_steps, best, model.state_dict(),
                    optimizer.state_dict(), logger.logdir)
