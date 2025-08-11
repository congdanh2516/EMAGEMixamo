import train
import os
import time
import csv
import sys
import warnings
import random
import numpy as np
import time
import pprint
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
import smplx

from utils import config, logger_tools, other_tools, metric
from utils import rotation_conversions as rc
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from scipy.spatial.transform import Rotation


class CustomTrainer(train.BaseTrainer):
    """
    motion representation learning
    """
    def __init__(self, args):
        super().__init__(args)

        ##--------------Copy from BEAT2022, ae_trainer.py------------##
        self.g_name = args.g_name
        self.pose_length = args.pose_length
        self.loss_meters = {
            'val_all': other_tools.AverageMeter('val_all'),
            'rec_val': other_tools.AverageMeter('rec_val'),
            'vel_val': other_tools.AverageMeter('vel_val'),
            'acceleration_val': other_tools.AverageMeter('acceleration_val'),
            
            'all': other_tools.AverageMeter('all'),
            'rec_loss': other_tools.AverageMeter('rec_loss'), 
            'vel_loss': other_tools.AverageMeter('vel_loss'),
            'acceleration_loss': other_tools.AverageMeter('acceleration_loss'),
        }
        self.best_epochs = {
            'val_all': [np.inf, 0],
            'rec_val': [np.inf, 0],
            'vel_val': [np.inf, 0],
            'acceleration_val': [np.inf, 0],
        }
        self.rec_loss = torch.nn.MSELoss(reduction='mean')
        self.vel_loss = torch.nn.MSELoss(reduction='mean') #torch.nn.L1Loss(reduction='mean')

        self.rec_weight = args.rec_weight
        self.vel_weight = args.vel_weight

    def train(self, epoch):
        self.model.train()
        t_start = time.time()
       
        ##--------------Copy from BEAT2022, ae_trainer.py------------##
        its_len = len(self.train_loader)
        ##--------------Copy from BEAT2022, ae_trainer.py------------##

        t_start = time.time()

        for its, dict_data in enumerate(self.train_loader):
            
            tar_exps = dict_data["facial"].cuda()
            t_data = time.time() - t_start
            
            ##--------------Copy from BEAT2022, ae_trainer.py------------##
             
            self.opt.zero_grad()
            loss = 0
            net_out = self.model(tar_exps)
            rec_exps = net_out["rec_pose"]

            ##-----------------Reconstruction loss------------##
            recon_loss = self.rec_loss(rec_exps, tar_exps) * self.args.rec_weight
         
            self.loss_meters['rec_loss'].update(recon_loss.item())
            
            loss += recon_loss

            velocity_loss =  self.vel_loss(rec_exps[:, 1:] - rec_exps[:, :-1], tar_exps[:, 1:] - tar_exps[:, :-1]) * self.args.rec_weight
            acceleration_loss =  self.vel_loss(rec_exps[:, 2:] + rec_exps[:, :-2] - 2 * rec_exps[:, 1:-1], tar_exps[:, 2:] + tar_exps[:, :-2] - 2 * tar_exps[:, 1:-1]) * self.args.rec_weight

            self.loss_meters['vel_loss'].update(velocity_loss.item())
            self.loss_meters['acceleration_loss'].update(acceleration_loss.item())
            
            loss += velocity_loss 
            loss += acceleration_loss 

            loss_embedding = net_out["embedding_loss"] # đã về số nhỏ => OK
            # print("loss_embedding shape:", loss_embedding.shape)
            # print("loss_embedding value:", loss_embedding)

            loss += loss_embedding

            self.loss_meters['all'].update(loss.item())
  
            
            loss.backward()
            if self.args.grad_norm != 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            if its % self.args.log_period == 0:
                self.recording(epoch, its, its_len, self.loss_meters, lr_g, 0, t_data, t_train, mem_cost)   
            if self.args.debug:
                if its == 1: break
        self.opt_s.step(epoch)
                    
    def val(self, epoch):
        self.model.eval()
        t_start = time.time()
        with torch.no_grad():
            its_len = len(self.val_loader)
            for its, dict_data in enumerate(self.val_loader):
                val_loss=0
                tar_exps = dict_data["facial"] # expressions
                tar_exps = tar_exps.cuda()         
                t_data = time.time() - t_start 

                net_out = self.model(tar_exps)
                rec_exps = net_out["rec_pose"]
    
                ##-----------------Reconstruction loss------------##
                recon_loss = self.rec_loss(rec_exps, tar_exps) * self.args.rec_weight
             
                self.loss_meters['rec_val'].update(recon_loss.item())
                
                val_loss += recon_loss
    
                velocity_loss =  self.vel_loss(rec_exps[:, 1:] - rec_exps[:, :-1], tar_exps[:, 1:] - tar_exps[:, :-1]) * self.args.rec_weight
                acceleration_loss =  self.vel_loss(rec_exps[:, 2:] + rec_exps[:, :-2] - 2 * rec_exps[:, 1:-1], tar_exps[:, 2:] + tar_exps[:, :-2] - 2 * tar_exps[:, 1:-1]) * self.args.rec_weight
    
                self.loss_meters['vel_val'].update(velocity_loss.item())
                self.loss_meters['acceleration_val'].update(acceleration_loss.item())
                
                val_loss += velocity_loss 
                val_loss += acceleration_loss 
    
                loss_embedding = net_out["embedding_loss"] # đã về số nhỏ => OK
    
                val_loss += loss_embedding
    
                self.loss_meters['val_all'].update(val_loss.item())
            
            self.val_recording(epoch, self.loss_meters)
            ##--------------Copy from BEAT2022, ae_trainer.py------------##
            
    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0

     
        test_seq_list = os.listdir(self.test_demo)
        test_seq_list.sort()

        self.model.eval()
        
        with torch.no_grad():

            if not os.path.exists(results_save_path):
                os.makedirs(results_save_path)
   
            for its, dict_data in enumerate(self.test_loader):
                tar_exps = dict_data["facial"]
                tar_exps = tar_exps.cuda()
                bs, n = tar_exps.shape[0], tar_exps.shape[1]
                remain = n%self.args.pose_length
                tar_exps = tar_exps[:, :n-remain, :]
                
                if True:
                    net_out = self.model(tar_exps)
                    rec_exps = net_out["rec_pose"]
                    n = rec_exps.shape[1]
                    rec_exps = rec_exps.cpu().numpy().reshape(bs*n, 51)
                else:
                    pass

                
                total_length += n 
                with open(f"{results_save_path}result_raw_{test_seq_list[its]}", 'w+') as f_real:
                    for line_id in range(rec_exps.shape[0]): #,args.pre_frames, args.pose_length
                        line_data = np.array2string(rec_exps[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                        f_real.write(line_data[1:-2]+'\n')
                        
            data_tools.result2target_vis(self.pose_version, results_save_path, results_save_path, self.test_demo, mode="upper_only", verbose=False)
            
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")