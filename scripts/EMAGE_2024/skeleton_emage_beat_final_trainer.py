import train
import os
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import pprint
from loguru import logger
from utils import rotation_conversions as rc
import smplx
from utils import config, logger_tools, other_tools, metric, data_transfer
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.data_tools import joints_list
import librosa

import lmdb as lmdb
import pickle
import shutil
import math
import textgrid as tg
import pandas as pd
import glob
import json
from termcolor import colored
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist
import pyarrow
from sklearn.preprocessing import normalize
import scipy.io.wavfile
from scipy import signal

class CustomTrainer(train.BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.joints = self.train_data.joints
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["skeleton_face"]
        self.tar_joint_list_upper = joints_list["skeleton_upper"]
        self.tar_joint_list_hands = joints_list["skeleton_hand"]
        self.tar_joint_list_lower = joints_list["skeleton_lower"]

        self.joints = 75
        
        self.joint_mask_face = np.zeros((len(list(self.ori_joint_list.keys())))*3)
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        self.joint_mask_upper = np.zeros((len(list(self.ori_joint_list.keys())))*3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
            
        self.joint_mask_hands = np.zeros((len(list(self.ori_joint_list.keys())))*3) 
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        self.joint_mask_lower = np.zeros((len(list(self.ori_joint_list.keys())))*3)
        for joint_name in self.tar_joint_list_lower:
            # if joint_name == "Hips":
            #     self.joint_mask_lower[3:6] = 1
            # else:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        self.tracker = other_tools.EpochTracker(["fid", "l1div", "bc", "rec", "trans", "vel", "transv", 'dis', 'gen', 'acc', 'transa', 'exp', 'lvd', 'mse', "cls", "rec_face", "latent", "cls_full", "cls_self", "cls_word", "latent_word","latent_self"], [False,True,True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False,False,False])
        
        
        vq_model_module = __import__(f"models.motion_representation", fromlist=["something"])
        self.args.vae_layer = 2
        self.args.vae_length = 256

        ## Face: 4, hands: 199, lower_foot: 632, upper: 399 
        ## Load Model for Upper
        self.args.vae_test_dim = 84
        self.vq_model_upper = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        # other_tools.load_checkpoints(self.vq_model_upper, "/data/nas07/PersonalData/thoai/beat_vicon_hips/pretrained/upper/last_399.bin", args.e_name)
        other_tools.load_checkpoints(self.vq_model_upper, "/work/vanthoai97/beat_vicon_hips/pretrained/upper/last_799.bin", args.e_name)
        

        ## Load model for Face
        self.args.vae_test_dim = 51
        self.vq_model_face = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        # print(self.vq_model_face)
        # other_tools.load_checkpoints(self.vq_model_face, "/data/nas07/PersonalData/thoai/beat_vicon_hips/pretrained/face/last_4.bin", args.e_name)
        other_tools.load_checkpoints(self.vq_model_face, "/work/vanthoai97/beat_vicon_hips/pretrained/face/last_457.bin", args.e_name)

        ## Load model for Hands
        self.args.vae_test_dim = 288
        self.vq_model_hands = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        # checkpoint = other_tools.load_checkpoints(self.vq_model_hands, "/data/nas07/PersonalData/thoai/beat_vicon_hips/pretrained/hands/last_199.bin", args.e_name)
        checkpoint = other_tools.load_checkpoints(self.vq_model_hands, "/work/vanthoai97/beat_vicon_hips/pretrained/hands/last_454.bin", args.e_name)

        ## Load model for Lower_foot
        self.args.vae_test_dim = 87
        self.args.vae_layer = 4
        self.vq_model_lower = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)        
        # other_tools.load_checkpoints(self.vq_model_lower, "/data/nas07/PersonalData/thoai/beat_vicon_hips/pretrained/lower_foot/last_632.bin", args.e_name)
        other_tools.load_checkpoints(self.vq_model_lower, "/work/vanthoai97/beat_vicon_hips/pretrained/lower_foot/last_114.bin", args.e_name)

        # Load model for global motion <=> Lower
        self.args.vae_test_dim = 87
        self.args.vae_layer = 4
        self.global_motion = getattr(vq_model_module, "VAEConvZero")(self.args).to(self.rank)
        # other_tools.load_checkpoints(self.global_motion, "/data/nas07/PersonalData/thoai/beat_vicon_hips/pretrained/lower_foot/last_632.bin", args.e_name)
        other_tools.load_checkpoints(self.vq_model_lower, "/work/vanthoai97/beat_vicon_hips/pretrained/lower_foot/last_114.bin", args.e_name)
        
        ## Global motion
        self.args.vae_test_dim = 450 # N.o joint nhân 6
        self.args.vae_layer = 4
        self.args.vae_length = 240

        self.vq_model_face.eval()
        self.vq_model_upper.eval()
        self.vq_model_hands.eval()
        self.vq_model_lower.eval()
        self.global_motion.eval()

        self.cls_loss = nn.NLLLoss().to(self.rank)
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        self.rec_loss = get_loss_func("GeodesicLoss").to(self.rank) 
        self.log_softmax = nn.LogSoftmax(dim=2).to(self.rank)

        self.loss_meters = {
        'val_all': other_tools.AverageMeter('val_all'),
        'rec_val': other_tools.AverageMeter('rec_val'),
        'vel_val': other_tools.AverageMeter('vel_val'),
        'acceleration_val': other_tools.AverageMeter('acceleration_val'),
        
    }
        self.best_epochs = {
            'val_all': [np.inf, 0],
            'rec_val': [np.inf, 0],
            'vel_val': [np.inf, 0],
            'acceleration_val': [np.inf, 0],
                            }
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 225)).cuda() # Số joint nhân 3
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t  
    
    def _load_data(self, dict_data):
        tar_pose_raw = dict_data["pose"]
        # print("tar_pose_raw shape:", tar_pose_raw.shape)
        tar_pose = tar_pose_raw[:, :, :225].to(self.rank) # tới số joint nhân 3
        # print("tar_pose shape:", tar_pose.shape)
        tar_contact = tar_pose_raw[:, :, 225:225+6].to(self.rank) # 6 giá trị tiếp theo của pose
        # print("tar_contact shape:", tar_contact.shape)
        tar_trans = dict_data["trans"].to(self.rank)
        # print("tar_trans shape:", tar_trans.shape)
        tar_exps = dict_data["facial"].to(self.rank) # torch.Size([bs, 64, 51])
        # print("tar_exps shape:", tar_exps.shape)
        in_audio = dict_data["audio"].to(self.rank)

        in_word = dict_data["word"].to(self.rank)
        # print("in_word max:", in_word.max().item())
        # print("in_word min:", in_word.min().item())
        # print("in_word shape:", in_word.shape)
        tar_id = dict_data["id"].to(self.rank).long()
        # print("speaker ids:", tar_id.min().item(), tar_id.max().item())
        tar_id = torch.where((tar_id >= 25) | (tar_id < 0), torch.tensor(0, device=tar_id.device), tar_id)

        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
        
        tar_pose_face = tar_exps
        
        tar_pose_hands = tar_pose[:, :, self.joint_mask_hands.astype(bool)] # ([bs, 64, 144])
        
        tar_pose_hands = rc.euler_angles_to_matrix(tar_pose_hands.reshape(bs, n, 48, 3), "XYZ")
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 48*6)
        # print("tar_pose_hands shape:", tar_pose_hands.shape)
        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)] # ([bs, 64, 42])
        tar_pose_upper = rc.euler_angles_to_matrix(tar_pose_upper.reshape(bs, n, 14, 3), "XYZ")
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 14*6)
        # print("tar_pose_upper shape:", tar_pose_upper.shape)
        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)] # [bs, 64, 39])
        tar_pose_leg = rc.euler_angles_to_matrix(tar_pose_leg.reshape(bs, n, 13, 3), "XYZ")
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 13*6)
        # print("tar_pose_leg shape:", tar_pose_leg.shape)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)
        # print("tar_pose_lower shape:", tar_pose_lower.shape)
        
        tar4dis = torch.cat([tar_pose_upper, tar_pose_hands, tar_pose_leg], dim=2)

        tar_index_value_face_top = self.vq_model_face.map2index(tar_pose_face) # bs*n/4
        tar_index_value_upper_top = self.vq_model_upper.map2index(tar_pose_upper) # bs*n/4
        tar_index_value_hands_top = self.vq_model_hands.map2index(tar_pose_hands) # bs*n/4
        tar_index_value_lower_top = self.vq_model_lower.map2index(tar_pose_lower) # bs*n/4

      
        latent_face_top = self.vq_model_face.map2latent(tar_pose_face) # bs*n/4
        # print("latent_face_top shape:", latent_face_top.shape)#([bs, 64, 256])
        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper) # bs*n/4
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands) # bs*n/4
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower) # bs*n/4
        
        latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2) # ([bs, 64, 768])
        # print("latent_in shape:", latent_in.shape)
        index_in = torch.stack([tar_index_value_upper_top, tar_index_value_hands_top, tar_index_value_lower_top], dim=-1).long() # ([bs, 64, 3])
        # print("index_in shape:", index_in.shape)
        
        tar_pose_6d =rc.euler_angles_to_matrix(tar_pose.reshape(bs, n, 75, 3), "XYZ")
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 75*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1) #([bs, 64, 459])
        # print("latent_all shape:", latent_all.shape)
        return {
            # "tar_pose_jaw": tar_pose_jaw,
            "tar_pose_face": tar_pose_face,
            "tar_pose_upper": tar_pose_upper,
            "tar_pose_lower": tar_pose_lower,
            "tar_pose_hands": tar_pose_hands,
            'tar_pose_leg': tar_pose_leg,
            "in_audio": in_audio,
            "in_word": in_word,
            "tar_trans": tar_trans,
            "tar_exps": tar_exps,
            # "tar_beta": tar_beta,
            "tar_pose": tar_pose,
            "tar4dis": tar4dis,
            "tar_index_value_face_top": tar_index_value_face_top,
            "tar_index_value_upper_top": tar_index_value_upper_top,
            "tar_index_value_hands_top": tar_index_value_hands_top,
            "tar_index_value_lower_top": tar_index_value_lower_top,
            "latent_face_top": latent_face_top,
            "latent_upper_top": latent_upper_top,
            "latent_hands_top": latent_hands_top,
            "latent_lower_top": latent_lower_top,
            "latent_in":  latent_in,
            "index_in": index_in,
            "tar_id": tar_id,
            "latent_all": latent_all,
            "tar_pose_6d": tar_pose_6d,
            "tar_contact": tar_contact,
        }

    
    def _g_training(self, loaded_data, use_adv, mode="train", epoch=0):
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        # ------ full generatation task ------ #
        mask_val = torch.ones(bs, n, self.args.pose_dims+3+6).float().cuda() ## 3 là trans, 6 là contact
        mask_val[:, :self.args.pre_frames, :] = 0.0
        
        net_out_val  = self.model(
            loaded_data['in_audio'], loaded_data['in_word'], mask=mask_val,
            in_id = loaded_data['tar_id'], in_motion = loaded_data['latent_all'],
            use_attentions = True)
        # print(f"MAGE_Transformer return:\n{net_out_val}")
        g_loss_final = 0
        loss_latent_face = self.reclatent_loss(net_out_val["rec_face"], loaded_data["latent_face_top"]) # reclatent_loss is MSELoss
        loss_latent_lower = self.reclatent_loss(net_out_val["rec_lower"], loaded_data["latent_lower_top"])
        loss_latent_hands = self.reclatent_loss(net_out_val["rec_hands"], loaded_data["latent_hands_top"])
        loss_latent_upper = self.reclatent_loss(net_out_val["rec_upper"], loaded_data["latent_upper_top"])
        loss_latent = self.args.lf*loss_latent_face + self.args.ll*loss_latent_lower + self.args.lh*loss_latent_hands + self.args.lu*loss_latent_upper
        self.tracker.update_meter("latent", "train", loss_latent.item())
        g_loss_final += loss_latent

        rec_index_face_val = self.log_softmax(net_out_val["cls_face"]).reshape(-1, self.args.vae_codebook_size) # [64, 256]
        rec_index_upper_val = self.log_softmax(net_out_val["cls_upper"]).reshape(-1, self.args.vae_codebook_size)
        rec_index_lower_val = self.log_softmax(net_out_val["cls_lower"]).reshape(-1, self.args.vae_codebook_size)
        rec_index_hands_val = self.log_softmax(net_out_val["cls_hands"]).reshape(-1, self.args.vae_codebook_size)
        tar_index_value_face_top = loaded_data["tar_index_value_face_top"].reshape(-1) # [4096]
        tar_index_value_upper_top = loaded_data["tar_index_value_upper_top"].reshape(-1)
        tar_index_value_lower_top = loaded_data["tar_index_value_lower_top"].reshape(-1)
        tar_index_value_hands_top = loaded_data["tar_index_value_hands_top"].reshape(-1) 
        # print(f"rec_index_face_val:{rec_index_face_val}, {rec_index_face_val.shape}")
        # print(f"tar_index_value_face_top: {tar_index_value_face_top}, {tar_index_value_face_top.shape}")
        # print(f"self.args.cf: {self.args.cf}")
        loss_cls = self.args.cf*self.cls_loss(rec_index_face_val, tar_index_value_face_top)\
            + self.args.cu*self.cls_loss(rec_index_upper_val, tar_index_value_upper_top)\
            + self.args.cl*self.cls_loss(rec_index_lower_val, tar_index_value_lower_top)\
            + self.args.ch*self.cls_loss(rec_index_hands_val, tar_index_value_hands_top)
        self.tracker.update_meter("cls_full", "train", loss_cls.item())
        g_loss_final += loss_cls 
        
        if mode == 'train':
            # logger.info("TRAIN")
        #     # ------ masked gesture moderling------ #
            mask_ratio = (epoch / self.args.epochs) * 0.95 + 0.05  
            mask = torch.rand(bs, n, self.args.pose_dims+3+6) < mask_ratio
            mask = mask.float().cuda()
            # print(f"loaded_data['latent_all'].shape: {loaded_data['latent_all'].shape}") # [8, 64, 225]
            net_out_self  = self.model(
                loaded_data['in_audio'], loaded_data['in_word'], mask=mask,
                in_id = loaded_data['tar_id'], in_motion = loaded_data['latent_all'],
                use_attentions = True, use_word=False)
            
            loss_latent_face_self = self.reclatent_loss(net_out_self["rec_face"], loaded_data["latent_face_top"])
            loss_latent_lower_self = self.reclatent_loss(net_out_self["rec_lower"], loaded_data["latent_lower_top"])
            loss_latent_hands_self = self.reclatent_loss(net_out_self["rec_hands"], loaded_data["latent_hands_top"])
            loss_latent_upper_self = self.reclatent_loss(net_out_self["rec_upper"], loaded_data["latent_upper_top"])
            loss_latent_self = self.args.lf*loss_latent_face_self + self.args.ll*loss_latent_lower_self + self.args.lh*loss_latent_hands_self + self.args.lu*loss_latent_upper_self
            self.tracker.update_meter("latent_self", "train", loss_latent_self.item())
            g_loss_final += loss_latent_self
            rec_index_face_self = self.log_softmax(net_out_self["cls_face"]).reshape(-1, self.args.vae_codebook_size)
            rec_index_upper_self = self.log_softmax(net_out_self["cls_upper"]).reshape(-1, self.args.vae_codebook_size)
            rec_index_lower_self = self.log_softmax(net_out_self["cls_lower"]).reshape(-1, self.args.vae_codebook_size)
            rec_index_hands_self = self.log_softmax(net_out_self["cls_hands"]).reshape(-1, self.args.vae_codebook_size)
            index_loss_top_self = self.cls_loss(rec_index_face_self, tar_index_value_face_top) + self.cls_loss(rec_index_upper_self, tar_index_value_upper_top) + self.cls_loss(rec_index_lower_self, tar_index_value_lower_top) + self.cls_loss(rec_index_hands_self, tar_index_value_hands_top)
            self.tracker.update_meter("cls_self", "train", index_loss_top_self.item())
            g_loss_final += index_loss_top_self
            # print("g_loss_final:", g_loss_final)
            # ------ masked audio gesture moderling ------ #
            net_out_word  = self.model(
                loaded_data['in_audio'], loaded_data['in_word'],
                mask=mask, in_id = loaded_data['tar_id'], 
                in_motion = loaded_data['latent_all'],
                use_attentions = True, use_word=True)

    
            loss_latent_face_word = self.reclatent_loss(net_out_word["rec_face"], loaded_data["latent_face_top"])
            loss_latent_lower_word = self.reclatent_loss(net_out_word["rec_lower"], loaded_data["latent_lower_top"])
            loss_latent_hands_word = self.reclatent_loss(net_out_word["rec_hands"], loaded_data["latent_hands_top"])
            loss_latent_upper_word = self.reclatent_loss(net_out_word["rec_upper"], loaded_data["latent_upper_top"])
            loss_latent_word = self.args.lf*loss_latent_face_word + self.args.ll*loss_latent_lower_word + self.args.lh*loss_latent_hands_word + self.args.lu*loss_latent_upper_word
            self.tracker.update_meter("latent_word", "train", loss_latent_word.item())
            g_loss_final += loss_latent_word

            rec_index_face_word = self.log_softmax(net_out_word["cls_face"]).reshape(-1, self.args.vae_codebook_size)
            rec_index_upper_word = self.log_softmax(net_out_word["cls_upper"]).reshape(-1, self.args.vae_codebook_size)
            rec_index_lower_word = self.log_softmax(net_out_word["cls_lower"]).reshape(-1, self.args.vae_codebook_size)
            rec_index_hands_word = self.log_softmax(net_out_word["cls_hands"]).reshape(-1, self.args.vae_codebook_size)
            index_loss_top_word = self.cls_loss(rec_index_face_word, tar_index_value_face_top) + self.cls_loss(rec_index_upper_word, tar_index_value_upper_top) + self.cls_loss(rec_index_lower_word, tar_index_value_lower_top) + self.cls_loss(rec_index_hands_word, tar_index_value_hands_top)
            self.tracker.update_meter("cls_word", "train", index_loss_top_word.item())
            g_loss_final += index_loss_top_word
            # print(f"g_loss_final: {g_loss_final.shape}")

        if mode != 'train':
            # logger.info("!TRAIN")
            if self.args.cu != 0:
                _, rec_index_upper = torch.max(rec_index_upper_val.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                rec_upper = self.vq_model_upper.decode(rec_index_upper)
            else:
                _, rec_index_upper, _, _ = self.vq_model_upper.quantizer(net_out_val["rec_upper"])
                rec_upper = self.vq_model_upper.decoder(rec_index_upper)
            if self.args.cl != 0:
                _, rec_index_lower = torch.max(rec_index_lower_val.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                rec_lower = self.vq_model_lower.decode(rec_index_lower)
            else:
                _, rec_index_lower, _, _ = self.vq_model_lower.quantizer(net_out_val["rec_lower"])
                rec_lower = self.vq_model_lower.decoder(rec_index_lower)
            if self.args.ch != 0:
                _, rec_index_hands = torch.max(rec_index_hands_val.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                rec_hands = self.vq_model_hands.decode(rec_index_hands)
            else:
                _, rec_index_hands, _, _ = self.vq_model_hands.quantizer(net_out_val["rec_hands"])
                rec_hands = self.vq_model_hands.decoder(rec_index_hands)
            if self.args.cf != 0:
                _, rec_index_face = torch.max(rec_index_face_val.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                rec_face = self.vq_model_face.decode(rec_index_face)
            else:
                _, rec_index_face, _, _ = self.vq_model_face.quantizer(net_out_val["rec_face"])
                rec_face = self.vq_model_face.decoder(rec_index_face)

            # rec_pose_jaw = rec_face[:, :, :6]
            rec_pose_legs = rec_lower[:, :, :78] # Số joint của lower nhân 6

            rec_pose_upper = rec_upper.reshape(bs, n, 14, 6)
            rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
            rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 14*3)
            rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
            
            rec_pose_lower = rec_pose_legs.reshape(bs, n, 13, 6)
            rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
            rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 13*3)
            rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
            rec_pose_hands = rec_hands.reshape(bs, n, 48, 6)
            rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
            rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 48*3)
            rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
            # rec_pose_jaw = rec_pose_jaw.reshape(bs*n, 6)
            # rec_pose_jaw = rc.rotation_6d_to_matrix(rec_pose_jaw)
            # rec_pose_jaw = rc.matrix_to_axis_angle(rec_pose_jaw).reshape(bs*n, 1*3)
            rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
            # rec_pose[:, 66:69] = rec_pose_jaw

            rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, j, 3))
            rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
            # print("rec_pose from __g_train__:", rec_pose.shape)

        if mode == 'train':
            return g_loss_final
        elif mode == 'val':
            return {
                'rec_pose': rec_pose,
                # rec_trans': rec_pose_trans,
                'tar_pose': loaded_data["tar_pose"],
            }
        else:
            return {
                'rec_pose': rec_pose,
                # 'rec_trans': rec_trans,
                'tar_pose': loaded_data["tar_pose"],
                'tar_exps': loaded_data["tar_exps"],
                # 'tar_beta': loaded_data["tar_beta"],
                'tar_trans': loaded_data["tar_trans"],
                # 'rec_exps': rec_exps,
            }
    

    def _g_test(self, loaded_data):
        mode = 'test'
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        tar_pose = loaded_data["tar_pose"]
        # tar_beta = loaded_data["tar_beta"]
        in_word = loaded_data["in_word"]
        tar_exps = loaded_data["tar_exps"]
        tar_contact = loaded_data["tar_contact"]
        in_audio = loaded_data["in_audio"]
        tar_trans = loaded_data["tar_trans"]

        remain = n%8 
        if remain != 0:
            tar_pose = tar_pose[:, :-remain, :]
            # tar_beta = tar_beta[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            in_word = in_word[:, :-remain]
            tar_exps = tar_exps[:, :-remain, :]
            tar_contact = tar_contact[:, :-remain, :]
            n = n - remain
        
        tar_pose_face = tar_exps
        tar_pose_hands = tar_pose[:, :, self.joint_mask_hands.astype(bool)]
        tar_pose_hands = rc.euler_angles_to_matrix(tar_pose_hands.reshape(bs, n, 48, 3), "XYZ")
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 48*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.euler_angles_to_matrix(tar_pose_upper.reshape(bs, n, 14, 3), "XYZ")
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 14*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.euler_angles_to_matrix(tar_pose_leg.reshape(bs, n, 13, 3), "XYZ")
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 13*6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)

        
        tar_pose_6d = rc.euler_angles_to_matrix(tar_pose.reshape(bs, n, 75, 3), "XYZ")
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 75*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
        
        rec_index_all_face = []
        rec_index_all_upper = []
        rec_index_all_lower = []
        rec_index_all_hands = []
        # rec_index_all_face_bot = []
        # rec_index_all_upper_bot = []
        # rec_index_all_lower_bot = []
        # rec_index_all_hands_bot = []
        
        roundt = (n - self.args.pre_frames) // (self.args.pose_length - self.args.pre_frames)
        remain = (n - self.args.pre_frames) % (self.args.pose_length - self.args.pre_frames)
        round_l = self.args.pose_length - self.args.pre_frames
        

        for i in range(0, roundt):
            in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            # audio fps is 16000 and pose fps is 30
            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames]
            in_id_tmp = loaded_data['tar_id'][:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+6).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0
            if i == 0:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]
            else:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]
                # print(latent_all_tmp.shape, latent_last.shape)
                latent_all_tmp[:, :self.args.pre_frames, :] = latent_last[:, -self.args.pre_frames:, :]
            
            net_out_val = self.model(
                in_audio = in_audio_tmp,
                in_word=in_word_tmp,
                mask=mask_val,
                in_id = in_id_tmp,
                in_motion = latent_all_tmp,
                use_attentions=True,)
            
            # print(f"net_out_val: {net_out_val}") # the result don't contain NaN
            
            if self.args.cu != 0:
                rec_index_upper = self.log_softmax(net_out_val["cls_upper"]).reshape(-1, self.args.vae_codebook_size)
                _, rec_index_upper = torch.max(rec_index_upper.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                #rec_upper = self.vq_model_upper.decode(rec_index_upper)
            else:
                _, rec_index_upper, _, _ = self.vq_model_upper.quantizer(net_out_val["rec_upper"])
                #rec_upper = self.vq_model_upper.decoder(rec_index_upper)
            if self.args.cl != 0:
                rec_index_lower = self.log_softmax(net_out_val["cls_lower"]).reshape(-1, self.args.vae_codebook_size)
                _, rec_index_lower = torch.max(rec_index_lower.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                #rec_lower = self.vq_model_lower.decode(rec_index_lower)
            else:
                _, rec_index_lower, _, _ = self.vq_model_lower.quantizer(net_out_val["rec_lower"])
                #rec_lower = self.vq_model_lower.decoder(rec_index_lower)
            if self.args.ch != 0:
                rec_index_hands = self.log_softmax(net_out_val["cls_hands"]).reshape(-1, self.args.vae_codebook_size)
                _, rec_index_hands = torch.max(rec_index_hands.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                #rec_hands = self.vq_model_hands.decode(rec_index_hands)
            else:
                _, rec_index_hands, _, _ = self.vq_model_hands.quantizer(net_out_val["rec_hands"])
                #rec_hands = self.vq_model_hands.decoder(rec_index_hands)
            if self.args.cf != 0:
                rec_index_face = self.log_softmax(net_out_val["cls_face"]).reshape(-1, self.args.vae_codebook_size)
                _, rec_index_face = torch.max(rec_index_face.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                #rec_face = self.vq_model_face.decoder(rec_index_face)
            else:
                _, rec_index_face, _, _ = self.vq_model_face.quantizer(net_out_val["rec_face"])
                #rec_face = self.vq_model_face.decoder(rec_index_face)

            if i == 0:
                rec_index_all_face.append(rec_index_face)
                rec_index_all_upper.append(rec_index_upper)
                rec_index_all_lower.append(rec_index_lower)
                rec_index_all_hands.append(rec_index_hands)
            else:
                rec_index_all_face.append(rec_index_face[:, self.args.pre_frames:])
                rec_index_all_upper.append(rec_index_upper[:, self.args.pre_frames:])
                rec_index_all_lower.append(rec_index_lower[:, self.args.pre_frames:])
                rec_index_all_hands.append(rec_index_hands[:, self.args.pre_frames:])

            if self.args.cu != 0:
                rec_upper_last = self.vq_model_upper.decode(rec_index_upper)
            else:
                rec_upper_last = self.vq_model_upper.decoder(rec_index_upper)
            if self.args.cl != 0:
                rec_lower_last = self.vq_model_lower.decode(rec_index_lower)
            else:
                rec_lower_last = self.vq_model_lower.decoder(rec_index_lower)
            if self.args.ch != 0:
                rec_hands_last = self.vq_model_hands.decode(rec_index_hands)
            else:
                rec_hands_last = self.vq_model_hands.decoder(rec_index_hands)


            ###
            rec_pose_legs = rec_lower_last[:, :, :78] # Số joint của lower nhân 6
            bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
            rec_pose_upper = rec_upper_last.reshape(bs, n, 14, 6)
            rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
            rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 14*3)
            rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
            
            rec_pose_lower = rec_pose_legs.reshape(bs, n, 13, 6)
            rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
            rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 13*3)
            rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
            
            rec_pose_hands = rec_hands_last.reshape(bs, n, 48, 6)
            rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
            rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 48*3)
            rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
            # rec_pose_jaw = rec_pose_jaw.reshape(bs*n, 6)
            # rec_pose_jaw = rc.rotation_6d_to_matrix(rec_pose_jaw)
            # rec_pose_jaw = rc.matrix_to_axis_angle(rec_pose_jaw).reshape(bs*n, 1*3)
            rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
            rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, j, 3))
            rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
            # print("rec_posse:", rec_pose.shape)
            ###
            rec_trans_v_s = rec_lower_last[:, :, 78:81]
            rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
            rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
            rec_y_trans = rec_trans_v_s[:,:,1:2]
            rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
            latent_last = torch.cat([rec_pose, rec_trans, rec_lower_last[:, :, 81:87]], dim=-1)
            # print("latent_last shape:", latent_last.shape)


        rec_index_face = torch.cat(rec_index_all_face, dim=1)
        rec_index_upper = torch.cat(rec_index_all_upper, dim=1)
        rec_index_lower = torch.cat(rec_index_all_lower, dim=1)
        rec_index_hands = torch.cat(rec_index_all_hands, dim=1)
        
        if self.args.cu != 0:
            rec_upper = self.vq_model_upper.decode(rec_index_upper)
        else:
            rec_upper = self.vq_model_upper.decoder(rec_index_upper)
        if self.args.cl != 0:
            rec_lower = self.vq_model_lower.decode(rec_index_lower)
        else:
            rec_lower = self.vq_model_lower.decoder(rec_index_lower)
        if self.args.ch != 0:
            rec_hands = self.vq_model_hands.decode(rec_index_hands)
        else:
            rec_hands = self.vq_model_hands.decoder(rec_index_hands)
        if self.args.cf != 0:
            rec_face = self.vq_model_face.decode(rec_index_face)
        else:
            rec_face = self.vq_model_face.decoder(rec_index_face)

###
        rec_exps = rec_face
        rec_pose_legs = rec_lower[:, :, :78] # Số joint của lower nhân 6
        bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 14, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 14*3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 13, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = rc.matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 13*6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 13*3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
        rec_pose_hands = rec_hands.reshape(bs, n, 48, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 48*3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
        # rec_pose_jaw = rec_pose_jaw.reshape(bs*n, 6)
        # rec_pose_jaw = rc.rotation_6d_to_matrix(rec_pose_jaw)
        # rec_pose_jaw = rc.matrix_to_axis_angle(rec_pose_jaw).reshape(bs*n, 1*3)
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 

        to_global = rec_lower
        to_global[:, :, 78:81] = 0.0
        to_global[:, :, :78] = rec_lower2global
        rec_global = self.global_motion(to_global)

        rec_trans_v_s = rec_global["rec_pose"][:, :, 78:81]
        rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
        rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
        rec_y_trans = rec_trans_v_s[:,:,1:2]
        rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        # tar_beta = tar_beta[:, :n, :]

        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs*n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
        tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs*n, j, 3))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
        
        return {
            'rec_pose': rec_pose,
            'rec_trans': rec_trans,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            # 'tar_beta': tar_beta,
            'tar_trans': tar_trans,
            'rec_exps': rec_exps,
        }

    
    def train(self, epoch):
        #torch.autograd.set_detect_anomaly(True)
        use_adv = bool(epoch>=self.args.no_adv_epoch)
        self.model.train()
        # self.d_model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, batch_data in enumerate(self.train_loader):
            loaded_data = self._load_data(batch_data)
            t_data = time.time() - t_start
    
            self.opt.zero_grad()
            g_loss_final = 0
            g_loss_final += self._g_training(loaded_data, use_adv, 'train', epoch)
            #with torch.autograd.detect_anomaly():
            g_loss_final.backward()
            if self.args.grad_norm != 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()
            
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            # lr_d = self.opt_d.param_groups[0]['lr']
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)   
            if self.args.debug:
                if its == 1: break
        self.opt_s.step(epoch)
        # self.opt_d_s.step(epoch) 
        
    
    def val(self, epoch):
        print("VAL WORKING")
        self.model.eval()
        # self.d_model.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.train_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_training(loaded_data, False, 'val', epoch)
                # print(f"net_out: {net_out}")
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']

                if (30/self.args.pose_fps) != 1:
                    assert 30%self.args.pose_fps == 0
                    n *= int(30/self.args.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                n = tar_pose.shape[1]
                
                remain = n%self.args.vae_test_len

                

                tar_pose = tar_pose[:, :n-remain, :]
                rec_pose = rec_pose[:, :n-remain, :]
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                
                tar_pose = rc.euler_angles_to_matrix(tar_pose.reshape(bs, n, j, 3), "XYZ") # ([32, 64, 12, 3, 3])
                # print("tar_pose euler_angles_to_matrix:", tar_pose.shape)
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6) # ([32, 64, 72])
                
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, n, j, 6))
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs, n, j, 6))
                
                
                if its == 0:
                    rec_pose_all = rec_pose
                    tar_pose_all = tar_pose
                else:
                    # rec_pose_all = np.concatenate([rec_pose_all, rec_pose], axis=1)                 
                    # tar_pose_all = np.concatenate([tar_pose_all, tar_pose], axis=1)
                    rec_pose_all = torch.cat([rec_pose_all, rec_pose], dim=1)
                    tar_pose_all = torch.cat([tar_pose_all, tar_pose], dim=1)
                    
                # print("    tar_pose_all:", tar_pose_all.shape)
                recon_loss = self.rec_loss(rec_pose_all, tar_pose_all)
                self.loss_meters['rec_val'].update(recon_loss.item()) 
                velocity_loss =  self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], tar_pose[:, 1:] - tar_pose[:, :-1])
                self.loss_meters['vel_val'].update(velocity_loss.item())
                acceleration_loss =  self.vel_loss(rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1], tar_pose[:, 2:] + tar_pose[:, :-2] - 2 * tar_pose[:, 1:-1])
                self.loss_meters['acceleration_val'].update(acceleration_loss.item())
                val_loss = recon_loss + velocity_loss + acceleration_loss
                self.loss_meters['val_all'].update(val_loss.item()) 
        
        self.tracker.update_meter("rec", "val", recon_loss)
        self.tracker.update_meter("vel", "val", velocity_loss)
        self.tracker.update_meter("acc", "val", acceleration_loss)

        self.val_recording(epoch, self.loss_meters)
        

    # """


    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        test_seq_list = os.listdir(self.test_demo)
        test_seq_list.sort()
        
        align = 0 
        latent_out = []
        latent_ori = []
        l2_all = 0 
        lvel = 0
        self.model.eval()
        # self.eval_copy.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data) 
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                # tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                # print(rec_pose.shape, tar_pose.shape)
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

                # if (30/self.args.pose_fps) != 1:
                #     assert 30%self.args.pose_fps == 0
                #     n *= int(30/self.args.pose_fps)
                
                tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
            
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_euler_angles(rec_pose, "XYZ").reshape(bs*n, j*3)
                # print("rec_pose final:", rec_pose.shape)
                rec_pose = np.rad2deg(rec_pose.cpu().numpy())
                trans= rec_trans.reshape(bs*n, 3).cpu().numpy()
                rec_pose = np.concatenate([trans, rec_pose], axis=1) # (frames*1, joint*3+3)

                total_length += n
                with open(f"{results_save_path}result_raw_{test_seq_list[its]}", 'w+') as f_real:
                    for line_id in range(rec_pose.shape[0]): #,args.pre_frames, args.pose_length
                        line_data = np.array2string(rec_pose[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                        f_real.write(line_data[1:-2]+'\n')
                        
            data_tools.result2target_vis(self.pose_version, results_save_path, results_save_path, self.test_demo,mode="all_or_lower", verbose=False)
            
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")
                    
