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
            'contact_val': other_tools.AverageMeter('contact_val'),
            'foot_val': other_tools.AverageMeter('foot_val'),
            
            'all': other_tools.AverageMeter('all'),
            'rec_loss': other_tools.AverageMeter('rec_loss'), 
            'vel_loss': other_tools.AverageMeter('vel_loss'),
            'acceleration_loss': other_tools.AverageMeter('acceleration_loss'),
            'contact_loss': other_tools.AverageMeter('contact_loss'),
            'foot_loss': other_tools.AverageMeter('foot_loss'),
        }
        self.best_epochs = {
            'val_all': [np.inf, 0],
            'rec_val': [np.inf, 0],
            'vel_val': [np.inf, 0],
            'acceleration_val': [np.inf, 0],
            'contact_val': [np.inf, 0],
            'foot_val': [np.inf, 0],
            
                            }

        self.rec_weight = args.rec_weight
        self.vel_weight = args.vel_weight
        self.args = args

        # Mới thêm vào
        self.joints = self.train_data.joints
        
        self.rec_loss = get_loss_func("GeodesicLoss")
        self.vel_loss = torch.nn.L1Loss(reduction='mean')
        self.vectices_loss = torch.nn.MSELoss(reduction='mean')

    def inverse_selection(self, filtered_t, selection_array, n):
        # 创建一个全为零的数组，形状为 n*165
        original_shape_t = np.zeros((n, selection_array.size))
        
        # 找到选择数组中为1的索引位置
        selected_indices = np.where(selection_array == 1)[0]
        
        # 将 filtered_t 的值填充到 original_shape_t 中相应的位置
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
            
        return original_shape_t

    def inverse_selection_tensor(self, filtered_t, selection_array, n):
    # 创建一个全为零的数组，形状为 n*165
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        
        # 找到选择数组中为1的索引位置
        selected_indices = torch.where(selection_array == 1)[0]
        
        # 将 filtered_t 的值填充到 original_shape_t 中相应的位置
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
            
        return original_shape_t
        

    
    def train(self, epoch):
        self.model.train()
        its_len = len(self.train_loader)
        t_start = time.time()
        

        for its, dict_data in enumerate(self.train_loader):
            
            tar_pose_raw = dict_data["pose"] ## 29 giá trị: 9 joints*3 + 3 (trans)
            # print("tar_pose_raw shape:", tar_pose_raw.shape)
            tar_trans = dict_data["trans"].cuda()

            tar_pose = tar_pose_raw[:, :, :27].cuda() 
            tar_contact = tar_pose_raw[:, :, 27:29].cuda() 
            
            bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
            tar_pose = rc.euler_angles_to_matrix(tar_pose.reshape(bs, n, j, 3), "YXZ") 
            tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6) 

            # tar_trans_vel_x = other_tools.estimate_linear_velocity(tar_trans[:, :, 0:1], dt=1/self.args.pose_fps)
            # tar_trans_vel_z = other_tools.estimate_linear_velocity(tar_trans[:, :, 2:3], dt=1/self.args.pose_fps)
            # tar_y_trans = tar_trans[:,:,1:2]
            # tar_xyz_trans = torch.cat([tar_x_trans, tar_y_trans, tar_z_trans], dim=-1)
            # tar_trans_copy = tar_xyz_trans

            
            tar_trans_copy = tar_trans
            tar_contact_copy = tar_contact
            
            in_tar_pose = torch.cat((tar_pose, tar_trans_copy, tar_contact_copy), dim=-1)
            # print("in_tar_pose shape:", in_tar_pose)
            t_data = time.time() - t_start     
            
            self.opt.zero_grad()
            loss=0
            
            net_out = self.model(in_tar_pose)
            rec_pose = net_out["rec_pose"][:, :, :j*6]
            
            rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs, n, j, 6)) 
            tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, n, j, 6)) 

            ##-----------------Reconstruction loss------------##
            recon_loss = self.rec_loss(rec_pose, tar_pose) * self.args.rec_weight * self.args.rec_pos_weight
            self.loss_meters['rec_loss'].update(recon_loss.item()) 
            loss += recon_loss

            ##-----------------Loss for contact------------##
            rec_contact = net_out["rec_pose"][:, :, j*6+3:j*6+3+2]
            loss_contact = self.vectices_loss(rec_contact, tar_contact) * self.args.rec_weight * self.args.rec_pos_weight
            self.loss_meters['contact_loss'].update(loss_contact.item()) 
            loss += loss_contact

            ##-----------------Velocity and acceleration loss------------##
            velocity_loss =  self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], tar_pose[:, 1:] - tar_pose[:, :-1]) * self.args.rec_weight
            acceleration_loss =  self.vel_loss(rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1], tar_pose[:, 2:] + tar_pose[:, :-2] - 2 * tar_pose[:, 1:-1]) * self.args.rec_weight

            self.loss_meters['vel_loss'].update(velocity_loss.item())
            self.loss_meters['acceleration_loss'].update(acceleration_loss.item())
            
            loss += velocity_loss 
            loss += acceleration_loss 



            model_contact = net_out["rec_pose"][:, :, j*6+3:j*6+7]

            ##--------------Loss for Position----------##
            # rec_trans = net_out["rec_pose"][:, :, j*6:j*6+3]
            # rec_x_trans = other_tools.velocity2position(rec_trans[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
            # rec_z_trans = other_tools.velocity2position(rec_trans[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
            # rec_y_trans = rec_trans[:,:,1:2]
            # rec_xyz_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
            # loss_trans_vel = self.vel_loss(rec_trans[:, :, 0:1], tar_trans_vel_x) * self.args.rec_weight \
            # + self.vel_loss(rec_trans[:, :, 2:3], tar_trans_vel_z) * self.args.rec_weight 

            ##--------------Loss for Foot-----------##
            model_contact = net_out["rec_pose"][:, :, j*6+3:j*6+3+2] 
            # print("model_contact:", model_contact.shape)
            # find static indices consistent with model's own predictions
            static_idx = model_contact > 0.95  # N x S x 4
            # print("static_idx:", static_idx.max().item())
            # print("static_idx shape:", static_idx.shape)
            num_true = static_idx.sum().item()
            # print("Number of True values:", num_true)

            foot_right = net_out["rec_pose"][:, :, 4*6:4*6+6].reshape(bs, n, 1, 6)   # joints 4 (based 0) (1 joints * 6 rotation6d)
            # print("foot_right", foot_right.shape)
            foot_left = net_out["rec_pose"][:, :, 8*6:8*6+6].reshape(bs, n, 1, 6)  # joints 8 (based 0 (1 joints * 6 rotation6d)
            # print("foot_left", foot_left.shape)

            model_feet = torch.tensor(torch.cat([foot_right, foot_left], axis=2)) # foot positions: NEED (N, S, 6, 3)
            model_feet = rc.rotation_6d_to_matrix(model_feet)
            
            # print("model_feet shape:", model_feet.shape)
            # model_feet = model_feet.reshape(bs, n, 6, 3)
            # print("model_feet shape:", model_feet.shape)
            model_foot_v = torch.zeros_like(model_feet)
            model_foot_v[:, :-1] = (
                model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
            )  # (N, S-1, 6, 3)
            # print("model_foot_v:", model_foot_v.shape)

            model_foot_v[~static_idx] = 0
            foot_loss = self.vel_loss(
                model_foot_v, torch.zeros_like(model_foot_v)
            )
            # print("foot_loss:", foot_loss)
            self.loss_meters['foot_loss'].update(foot_loss.item()*self.args.rec_weight * self.args.rec_ver_weight*20)
            
            loss += foot_loss*self.args.rec_weight*self.args.rec_ver_weight*20 
            
            ## Embedding loss
            loss_embedding = net_out["embedding_loss"]
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

                tar_pose_raw = dict_data["pose"]
                tar_trans = dict_data["trans"].cuda()
    
                tar_pose = tar_pose_raw[:, :, :27].cuda() 
                tar_contact = tar_pose_raw[:, :, 27:29].cuda() 

                
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                tar_pose = rc.euler_angles_to_matrix(tar_pose.reshape(bs, n, j, 3), "YXZ") 
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6) 

                
                tar_trans_copy = tar_trans
                tar_contact_copy = tar_contact
                
                in_tar_pose = torch.cat((tar_pose, tar_trans_copy, tar_contact_copy), dim=-1)
                t_data = time.time() - t_start     
                
                self.opt.zero_grad()
                val_loss=0
                
                net_out = self.model(in_tar_pose)
                rec_pose = net_out["rec_pose"][:, :, :j*6]
                
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs, n, j, 6)) 
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, n, j, 6)) 
    
                ##-----------------Reconstruction loss------------##
                recon_loss = self.rec_loss(rec_pose, tar_pose) * self.args.rec_weight * self.args.rec_pos_weight
                self.loss_meters['rec_val'].update(recon_loss.item()) 
                val_loss += recon_loss
    
                ##-----------------Loss for contact------------##
                rec_contact = net_out["rec_pose"][:, :, j*6+3:j*6+3+2]
                loss_contact = self.vectices_loss(rec_contact, tar_contact) * self.args.rec_weight * self.args.rec_pos_weight
                self.loss_meters['contact_val'].update(loss_contact.item()) 
                val_loss += loss_contact
    
                ##-----------------Velocity and acceleration loss------------##
                velocity_loss =  self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], tar_pose[:, 1:] - tar_pose[:, :-1]) * self.args.rec_weight
                acceleration_loss =  self.vel_loss(rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1], tar_pose[:, 2:] + tar_pose[:, :-2] - 2 * tar_pose[:, 1:-1]) * self.args.rec_weight
    
                self.loss_meters['vel_val'].update(velocity_loss.item())
                self.loss_meters['acceleration_val'].update(acceleration_loss.item())
                
                val_loss += velocity_loss 
                val_loss += acceleration_loss 
    
                ##--------------Loss for Position----------##
                # rec_trans = net_out["rec_pose"][:, :, j*6:j*6+3]
                # rec_x_trans = other_tools.velocity2position(rec_trans[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
                # rec_z_trans = other_tools.velocity2position(rec_trans[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
                # rec_y_trans = rec_trans[:,:,1:2]
                # rec_xyz_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
                # loss_trans_vel = self.vel_loss(rec_trans[:, :, 0:1], tar_trans_vel_x) * self.args.rec_weight \
                # + self.vel_loss(rec_trans[:, :, 2:3], tar_trans_vel_z) * self.args.rec_weight 
    
                ##--------------Loss for Foot-----------##
                model_contact = net_out["rec_pose"][:, :, j*6+3:j*6+3+2]
                # print("model_contact:", model_contact.shape)
                # find static indices consistent with model's own predictions
                static_idx = model_contact > 0.95  # N x S x 4
                # print("static_idx:", static_idx)
    
                foot_right = net_out["rec_pose"][:, :, 4*6:4*6+6].reshape(bs, n, 1, 6)   # joints 4 
                # print("foot_right", foot_right.shape)
                foot_left = net_out["rec_pose"][:, :, 8*6:8*6+6].reshape(bs, n, 1, 6)  # joints 8
                # print("foot_left", foot_left.shape)
    
                model_feet = torch.tensor(torch.cat([foot_right, foot_left], axis=2)) # foot positions: NEED (N, S, 6, 3)
                model_feet = rc.rotation_6d_to_matrix(model_feet)
            
                # print("model_feet shape:", model_feet.shape)
                model_foot_v = torch.zeros_like(model_feet)
                model_foot_v[:, :-1] = (
                    model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
                )  # (N, S-1, 6, 3)
                # print("model_foot_v:", model_foot_v)
    
                model_foot_v[~static_idx] = 0
                foot_loss = self.vel_loss(
                    model_foot_v, torch.zeros_like(model_foot_v)
                )
                # print("foot_loss:", foot_loss)
                self.loss_meters['foot_val'].update(foot_loss.item()*self.args.rec_weight * self.args.rec_ver_weight*20)
                
                val_loss += foot_loss*self.args.rec_weight*self.args.rec_ver_weight*20 
                
                ## Embedding loss
                loss_embedding = net_out["embedding_loss"]
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
                tar_pose_raw = dict_data["pose"]
                tar_trans = dict_data["trans"].cuda()

                tar_pose = tar_pose_raw[:, :, :27].cuda() 
                tar_contact = tar_pose_raw[:, :, 27:29].cuda() 
                
                
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                
                tar_pose = rc.euler_angles_to_matrix(tar_pose.reshape(bs, n, j, 3), "XYZ")
                    # print("tar_pose euler_angles_to_matrix:", tar_pose.shape)
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)

                remain = n%self.args.pose_length
                tar_pose = tar_pose[:, :n-remain, :]
                tar_contact = tar_contact[:, :n-remain, :]
                tar_trans_copy = tar_trans[:, :n-remain, :]
                tar_contact_copy = tar_contact
                in_tar_pose = torch.cat([tar_pose, tar_trans_copy, tar_contact_copy], dim=-1)
                
                if True:
                    net_out = self.model(in_tar_pose)
                    rec_pose = net_out["rec_pose"][:, :, :j*6]
                    n = rec_pose.shape[1]
                    
                    rec_pose = rec_pose.reshape(bs, n, j, 6) 
                    rec_trans = net_out["rec_pose"][:, :, j*6:j*6+3] # Trong code gốc lấy 2 cái trừ nhau cho bằng 0
                    # print("rec_trans:", rec_trans.shape)

                    rec_pose = rc.rotation_6d_to_matrix(rec_pose)
                    rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                    rec_pose = rec_pose.cpu().numpy()

                    ##----------Comment nếu muốn dùng trans từ model-----------##
                    # rec_x_trans = other_tools.velocity2position(rec_trans[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
                    # rec_z_trans = other_tools.velocity2position(rec_trans[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
                    # rec_y_trans = rec_trans[:,:,1:2]
                    # rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
                    ##---------------------------------------------------------##
                else:
                    pass

                rec_pose = rc.axis_angle_to_matrix(torch.from_numpy(rec_pose.reshape(bs*n, j, 3)))
                rec_pose = np.rad2deg(rc.matrix_to_euler_angles(rec_pose, "YXZ")).reshape(bs*n, j*3).numpy()  
                trans= rec_trans.reshape(bs*n, 3).cpu().numpy()

                rec_pose = np.concatenate([trans, rec_pose], axis=1) # (frames*1, joint*3+3)
                # rec_pose này sẽ được viết vào result_raw
                # print("rec_pose final:", rec_pose)

                ## res_bvh bắt đầu ghi từ frame thứ 2 của rec_pose
                total_length += n 
                with open(f"{results_save_path}result_raw_{test_seq_list[its]}", 'w+') as f_real:
                    for line_id in range(rec_pose.shape[0]): #,args.pre_frames, args.pose_length
                        line_data = np.array2string(rec_pose[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                        f_real.write(line_data[1:-2]+'\n')
                        
            data_tools.result2target_vis(self.pose_version, results_save_path, results_save_path, self.test_demo, mode="all_or_lower", verbose=False)
            
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")