import os
import signal
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
import time
import pprint
from loguru import logger
import smplx
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt
from utils import config, logger_tools, other_tools, metric
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.rank = dist.get_rank()
        self.checkpoint_path = args.out_path + "custom/" + args.name + args.notes + "/" #wandb.run.dir #args.cache_path+args.out_path+"/"+args.name
        if self.rank==0:
            if self.args.stat == "ts":
                self.writer = SummaryWriter(log_dir=args.out_path + "custom/" + args.name + args.notes + "/")
            else:
                wandb.init(project=args.project, entity="liu1997", dir=args.out_path, name=args.name[12:] + args.notes)
                wandb.config.update(args)
                self.writer = None  
        #self.test_demo = args.data_path + args.test_data_path + "bvh_full/"        
        # self.train_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "train")
        # self.train_loader = torch.utils.data.DataLoader(
        #     self.train_data, 
        #     batch_size=args.batch_size,  
        #     shuffle=False if args.ddp else True,  
        #     num_workers=args.loader_workers,
        #     drop_last=True,
        #     sampler=torch.utils.data.distributed.DistributedSampler(self.train_data) if args.ddp else None, 
        # )
        # self.train_length = len(self.train_loader)
        # logger.info(f"Init train dataloader success")
       
        # self.val_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "val")  
        # self.val_loader = torch.utils.data.DataLoader(
        #     self.val_data, 
        #     batch_size=args.batch_size,  
        #     shuffle=False,  
        #     num_workers=args.loader_workers,
        #     drop_last=False,
        #     sampler=torch.utils.data.distributed.DistributedSampler(self.val_data) if args.ddp else None, 
        # )
        # logger.info(f"Init val dataloader success")
        if self.rank == 0:
            self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
            self.test_loader = torch.utils.data.DataLoader(
                self.test_data, 
                batch_size=1,  
                shuffle=False,  
                num_workers=args.loader_workers,
                drop_last=False,
            )
        logger.info(f"Init test dataloader success")
        model_module = __import__(f"models.{args.model}", fromlist=["something"])
        
        if args.ddp:
            self.model = getattr(model_module, args.g_name)(args).to(self.rank)
            process_group = torch.distributed.new_group()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)   
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank,
                             broadcast_buffers=False, find_unused_parameters=False)
        else: 
            self.model = torch.nn.DataParallel(getattr(model_module, args.g_name)(args), args.gpus).cuda()
        
        if self.rank == 0:
            logger.info(self.model)
            logger.info(f"init {args.g_name} success")
            if args.stat == "wandb":
                wandb.watch(self.model)

@logger.catch
def main_worker(rank, world_size, args):
    #os.environ['TRANSFORMERS_CACHE'] = args.data_path_1 + "hub/"
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args)
      
    # return one intance of trainer
    trainer = __import__(f"{args.trainer}_trainer", fromlist=["something"]).CustomTrainer(args) if args.trainer != "base" else BaseTrainer(args) 
    other_tools.load_checkpoints(trainer.model, args.test_ckpt, args.g_name)
    trainer.test(999)
    
    
            
if __name__ == "__main__":
    os.environ["MASTER_ADDR"]='127.0.0.1'
    os.environ["MASTER_PORT"]='8675'
    #os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    args = config.parse_args()
    if args.ddp:
        mp.set_start_method("spawn", force=True)
        mp.spawn(
            main_worker,
            args=(len(args.gpus), args,),
            nprocs=len(args.gpus),
                )
    else:
        main_worker(0, 1, args)