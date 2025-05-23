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
        self.ddp = args.ddp
        self.gpus = len(args.gpus)
        self.pose_version = args.pose_version
        print("gpus:", self.gpus)

        # add new ----- begin
        self.best_epochs = {
            'fid_val': [np.inf, 0],
            'rec_val': [np.inf, 0],
        }
        # add new ----- end 
        
        self.checkpoint_path = args.out_path + "custom/" + args.name + args.notes + "/" #wandb.run.dir #args.cache_path+args.out_path+"/"+args.name
        if self.rank==0:
            if self.args.stat == "ts":
                logger.info("Initializing W&B with stat=ts...")
                self.writer = SummaryWriter(log_dir=args.out_path + "custom/" + args.name + args.notes + "/")
            else:
        
                logger.info("Initializing W&B...")
                # print("\n\n\n\n\n", args.project, "\n\n\n\n\n")
                wandb.init(project=args.project, entity="caocongdanhccd-national-central-university", dir=args.out_path, name=args.name[12:] + args.notes)
                # , name=args.name[12:] + args.notes
                wandb.config.update(args)
                self.writer = None
                
        # direct to test data
        self.test_demo = args.root_path + args.test_data_path + "bvh_full/"  
        # module name, fromlist: accept any value, let python understand that it is importing the child module and not the parent module, if fromlist is null, python will import dataloaders package
        # 
        self.train_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "train")

        print(self.train_data[0]['pose'].shape)

        # DataLoader is a class in PyTorch creating data loader
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, 
            batch_size=args.batch_size,  
            # ddp - Distributed Data Parallel
            shuffle=False if args.ddp else True,  
            # Determine the number of child processes used to load data
            num_workers=args.loader_workers,
            # Skip the last batch if the number of samples is not divisible by the batch size.
            drop_last=True,
            # DistributedSampler is used to distribute data to different GPUs, is a technique for training models on multiple GPUs
            sampler=torch.utils.data.distributed.DistributedSampler(self.train_data) if args.ddp else None, 
        )
        # number of batches per epoch
        # len() is used to calculate the number of bactches in this DataLoader, cho biết có bao nhiêu lần chúng ta cần lặp qua DataLoader để xử lý hết toàn bộ dữ liệu huấn luyện
        self.train_length = len(self.train_loader)
        logger.info(f"Init train dataloader success")

        # respectively, custom for validation data
        self.val_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "val")  
        print(self.val_data[0]['pose'].shape)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_data, 
            batch_size=args.batch_size,  
            shuffle=False,  
            num_workers=args.loader_workers,
            drop_last=False,
            sampler=torch.utils.data.distributed.DistributedSampler(self.val_data) if args.ddp else None, 
        )
        logger.info(f"Init val dataloader success")

        # rank is the rank of the current process
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
        
        if args.d_name is not None:
            if args.ddp:
                self.d_model = getattr(model_module, args.d_name)(args).to(self.rank)
                self.d_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.d_model, process_group)   
                self.d_model = DDP(self.d_model, device_ids=[self.rank], output_device=self.rank, 
                                   broadcast_buffers=False, find_unused_parameters=False)
            else:    
                self.d_model = torch.nn.DataParallel(getattr(model_module, args.d_name)(args), args.gpus).cuda()
            if self.rank == 0:
                logger.info(self.d_model)
                logger.info(f"init {args.d_name} success")
                if args.stat == "wandb":
                    wandb.watch(self.d_model)
            self.opt_d = create_optimizer(args, self.d_model, lr_weight=args.d_lr_weight)
            self.opt_d_s = create_scheduler(args, self.opt_d)
           
        if args.e_name is not None:
            """
            bugs on DDP training using eval_model, using additional eval_copy for evaluation 
            """
            eval_model_module = __import__(f"models.{args.eval_model}", fromlist=["something"])
            # eval copy is for single card evaluation
            if self.args.ddp:
                self.eval_model = getattr(eval_model_module, args.e_name)(args).to(self.rank)
                self.eval_copy = getattr(eval_model_module, args.e_name)(args).to(self.rank) 
            else:
                self.eval_model = getattr(eval_model_module, args.e_name)(args)
                # print(f"self.eval_model: {self.eval_model}")
                self.eval_copy = getattr(eval_model_module, args.e_name)(args).to(self.rank)
                
            #if self.rank == 0:
            # /data/nas07/PersonalData/danh/PantoMatrix/beat_4english_30_full/ae_300_all_data_v0.2.1.bin
            other_tools.load_checkpoints(self.eval_copy, "./BEAT2/beat_english_v2.0.0/weights/AESKConv_240_100.bin", args.e_name)
            other_tools.load_checkpoints(self.eval_model, "./BEAT2/beat_english_v2.0.0/weights/AESKConv_240_100.bin", args.e_name)
            if self.args.ddp:
                self.eval_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.eval_model, process_group)   
                self.eval_model = DDP(self.eval_model, device_ids=[self.rank], output_device=self.rank,
                                      broadcast_buffers=False, find_unused_parameters=False)
            self.eval_model.eval()
            self.eval_copy.eval()
            if self.rank == 0:
                logger.info(self.eval_model)
                logger.info(f"init {args.e_name} success")  
                if args.stat == "wandb":
                    wandb.watch(self.eval_model) 
        self.opt = create_optimizer(args, self.model)
        self.opt_s = create_scheduler(args, self.opt)
        

        self.l1_calculator = metric.L1div() if self.rank == 0 else None
       
    
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t

    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        if len(filtered_t.shape) == 2:
            original_shape_t = torch.zeros((n, 165)).cuda()
            for i in range(n):
                original_shape_t[i, selected_indices] = filtered_t[i]
        elif len(filtered_t.shape) == 3:
            bs, n, _ = filtered_t.shape
            original_shape_t = torch.zeros((bs, n, 165), device='cuda')
            expanded_indices = selected_indices.unsqueeze(0).unsqueeze(0).expand(bs, n, -1)
            original_shape_t.scatter_(2, expanded_indices, filtered_t)
        return original_shape_t

    def inverse_selection_tensor_6d(self, filtered_t, selection_array, n):
        new_selected_array = np.zeros((330))
        new_selected_array[::2] = selection_array
        new_selected_array[1::2] = selection_array 
        selection_array = new_selected_array
        selection_array = torch.from_numpy(selection_array).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        if len(filtered_t.shape) == 2:
            original_shape_t = torch.zeros((n, 330)).cuda()
            for i in range(n):
                original_shape_t[i, selected_indices] = filtered_t[i]
        elif len(filtered_t.shape) == 3:
            bs, n, _ = filtered_t.shape
            original_shape_t = torch.zeros((bs, n, 330), device='cuda')
            expanded_indices = selected_indices.unsqueeze(0).unsqueeze(0).expand(bs, n, -1)
            original_shape_t.scatter_(2, expanded_indices, filtered_t)
        return original_shape_t

    ## Code from BEAT
    def recording(self, epoch, its, its_len, loss_meters, lr_g, lr_d, t_data, t_train, mem_cost):
        if self.rank == 0:
            pstr = "[%03d][%03d/%03d]  "%(epoch, its, its_len)
            for name, loss_meter in self.loss_meters.items():
                if "val" not in name:
                    if loss_meter.count > 0:
                        pstr += "{}: {:.3f}\t".format(loss_meter.name, loss_meter.avg)
                        wandb.log({loss_meter.name: loss_meter.avg}, step=epoch*self.train_length+its)
                        loss_meter.reset()
            pstr += "glr: {:.1e}\t".format(lr_g)
            pstr += "dlr: {:.1e}\t".format(lr_d)
            wandb.log({'glr': lr_g, 'dlr': lr_d}, step=epoch*self.train_length+its)
            pstr += "dtime: %04d\t"%(t_data*1000)        
            pstr += "ntime: %04d\t"%(t_train*1000)
            pstr += "mem: {:.2f} ".format(mem_cost*self.gpus)
            logger.info(pstr)


    def train_recording(self, epoch, its, t_data, t_train, mem_cost, lr_g, lr_d=None):
        pstr = "[%03d][%03d/%03d]  "%(epoch, its, self.train_length)
        for name, states in self.tracker.loss_meters.items():
            metric = states['train']
            if metric.count > 0:
                pstr += "{}: {:.3f}\t".format(name, metric.avg)
                self.writer.add_scalar(f"train/{name}", metric.avg, epoch*self.train_length+its) if self.args.stat == "ts" else wandb.log({name: metric.avg}, step=epoch*self.train_length+its)
        pstr += "glr: {:.1e}\t".format(lr_g)
        self.writer.add_scalar("lr/glr", lr_g, epoch*self.train_length+its) if self.args.stat == "ts" else wandb.log({'glr': lr_g}, step=epoch*self.train_length+its)
        if lr_d is not None:
            pstr += "dlr: {:.1e}\t".format(lr_d)
            self.writer.add_scalar("lr/dlr", lr_d, epoch*self.train_length+its) if self.args.stat == "ts" else wandb.log({'dlr': lr_d}, step=epoch*self.train_length+its)
        pstr += "dtime: %04d\t"%(t_data*1000)        
        pstr += "ntime: %04d\t"%(t_train*1000)
        pstr += "mem: {:.2f} ".format(mem_cost*len(self.args.gpus))
        logger.info(pstr)
        
## Code from BEAT
    def val_recording(self, epoch, metrics):
        if self.rank == 0: 
            pstr_curr = "Curr info >>>>  "
            pstr_best = "Best info >>>>  "

            # print(f"Metrics: {metrics}")
            # print(f"\nMetrics: {metrics.items()}")
            
            for name, metric in metrics.items():
                # print(f"name: {name}")
                if "val" in name:
                    if metric.count > 0:
                        # print(f"metric.count: {metric.count}")
                        pstr_curr += "{}: {:.3f}     \t".format(metric.name, metric.avg)
                        # print(f"metric name/avg: {metric.name}, {metric.avg}") # rec_val, 16.447102122836643
                        wandb.log({metric.name: metric.avg}, step=epoch*self.train_length)
                        # print(f"self.best_epochs: {self.best_epochs}")
                        if metric.avg < self.best_epochs[metric.name][0]:
                            self.best_epochs[metric.name][0] = metric.avg
                            self.best_epochs[metric.name][1] = epoch
                            other_tools.save_checkpoints(os.path.join(self.checkpoint_path, f"{metric.name}.bin"), self.model, opt=None, epoch=None, lrs=None)        
                        metric.reset()
            for k, v in self.best_epochs.items(): # v[0]: fid_val, rec_val
                pstr_best += "{}: {:.3f}({:03d})\t".format(k, v[0], v[1])
                # print(f"l, v[0], v[1]: {k}, {v[0]}, {v[1]}")
                # print(pstr_best)
            logger.info(f"{pstr_curr}") # Curr info >>>>
            logger.info(f"{pstr_best}") # Best info >>>>
   
    def test_recording(self, dict_name, value, epoch):
        self.tracker.update_meter(dict_name, "test", value)
        _ = self.tracker.update_values(dict_name, 'test', epoch)

@logger.catch
def main_worker(rank, world_size, args):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args)
      
    # return one intance of trainer
    trainer = __import__(f"{args.trainer}_trainer", fromlist=["something"]).CustomTrainer(args) if args.trainer != "base" else BaseTrainer(args) 

    logger.info("Training from starch ...")          
    start_time = time.time()
    for epoch in range(args.epochs):
        logger.info(f"NUMBER OF EPOCH: {args.epochs}")
        # trainer.test(epoch)

        # print(f"EPOCH: {epoch}")

        if trainer.ddp: trainer.val_loader.sampler.set_epoch(epoch)
        # print(f"self.eval_model: {self.eval_model}")
        # d
        # print(f"epoch: {epoch}")
        # print("---------")
        trainer.val(epoch)
        # print("---------")
        epoch_time = time.time()-start_time
        if trainer.rank == 0: logger.info("Time info >>>>  elapsed: %.2f mins\t"%(epoch_time/60)+"remain: %.2f mins"%((args.epochs/(epoch+1e-7)-1)*epoch_time/60))
        if trainer.ddp: trainer.train_loader.sampler.set_epoch(epoch)
        trainer.train(epoch) 
        if (epoch+1) % args.test_period == 0:
            if rank == 0:
                trainer.test(epoch)
                other_tools.save_checkpoints(os.path.join(trainer.checkpoint_path, f"last_{epoch}.bin"), trainer.model, opt=None, epoch=None, lrs=None)
            
    for k, v in trainer.best_epochs.items():
        wandb.log({f"{k}_best": v[0], f"{k}_epoch": v[1]})
    
    if rank == 0:
        wandb.finish()


    
            
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