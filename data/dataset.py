import torch
from torchvision import transforms

from torchmeta.transforms import ClassSplitter, Categorical

from data.income import Income
from data.diabetes import Diabetes
from data.dna import DNA  

def get_meta_dataset(P, dataset, r1=None, r2=None, only_test=False):

    if dataset == 'income':
        meta_train_dataset = Income(
            tabular_size=105,
            seed=P.seed,
            source='train',
            shot=P.num_shots,
            tasks_per_batch=P.batch_size,
            test_num_way=P.num_ways,
            query=P.num_shots_test,
            r1=r1,  
            r2=r2   
        )

        meta_val_dataset = Income(
            tabular_size=105,
            seed=P.seed,
            source='val',
            shot=1,
            tasks_per_batch=P.test_batch_size,
            test_num_way=2,
            query=30,
            r1=r1,  
            r2=r2   
        )

    elif dataset == 'diabetes':
        meta_train_dataset = Diabetes(
            tabular_size=8,  
            seed=P.seed,
            source='train',
            shot=P.num_shots,
            tasks_per_batch=P.batch_size,
            test_num_way=P.num_ways,
            query=P.num_shots_test,
            r1=r1,  
            r2=r2   
        )

        meta_val_dataset = Diabetes(
            tabular_size=8,
            seed=P.seed,
            source='val',
            shot=1,
            tasks_per_batch=P.test_batch_size,
            test_num_way=2,
            query=30,
            r1=r1,  
            r2=r2   
        )

    elif dataset == 'dna':
        meta_train_dataset = DNA(
            tabular_size=360, 
            seed=P.seed,
            source='train',
            shot=P.num_shots,
            tasks_per_batch=P.batch_size,
            test_num_way=P.num_ways,
            query=P.num_shots_test,
            r1=r1,  
            r2=r2 
        )

        meta_val_dataset = DNA(
            tabular_size=360,
            seed=P.seed,
            source='val',
            shot=1,
            tasks_per_batch=P.test_batch_size,
            test_num_way=2,
            query=30,
            r1=r1, 
            r2=r2  
        )
  
    else:
        raise NotImplementedError()

    return meta_train_dataset, meta_val_dataset
