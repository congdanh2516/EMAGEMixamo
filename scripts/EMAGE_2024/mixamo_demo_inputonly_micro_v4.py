import numpy as np
import sounddevice as sd
import threading
import time

import os
import signal
import time
import csv
import sys
import warnings
import random
import gradio as gr
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
# import wandb
import matplotlib.pyplot as plt
from utils import config, logger_tools, other_tools, other_tools_hf, metric, data_transfer
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.data_tools import joints_list
from utils import rotation_conversions as rc
import soundfile as sf
import librosa 
import tempfile
import scipy.io.wavfile as wavfile
import json

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


_global_inference = False
inference_model = None

_vq_model_face = None
_vq_model_upper = None
_vq_model_hands = None
_vq_model_lower = None
_global_motion = None

_joint_mask_face = None
_joint_mask_upper = None
_joint_mask_hands = None 
_joint_mask_lower = None

_joints = None

_log_softmax = None

current_dir = os.path.dirname(os.path.abspath(__file__))
pretrained_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "pretrained/mixamo"))

test_checkpoint = pretrained_dir + "/last_92.bin"

_test_loader = None

def fix_abnormal_joint_motion_soft(rec_pose, joint_idx=10, fps=30.0,
                                   abnormal_threshold_deg_per_s=800.0,
                                   blend_factor=0.2):
    """
    Soft-fixes abnormal angular velocity for a given joint in Euler angle sequences.
    If abnormal, moves the current frame's joint rotation toward the previous frame's
    by a blend_factor instead of replacing it completely.

    Parameters
    ----------
    rec_pose : np.ndarray
        Shape: (bs, n, j, 3) - Euler angles in degrees.
    joint_idx : int
        Index of the joint to check (0-based). Default = 10 for the 11th joint.
    fps : float
        Frames per second of the motion data.
    abnormal_threshold_deg_per_s : float
        Velocity threshold in deg/s above which frames are considered abnormal.
    blend_factor : float
        How much to move toward the previous frame when abnormal.
        0.0 = no change, 1.0 = freeze completely to previous frame.

    Returns
    -------
    np.ndarray
        Modified rec_pose array with abnormal frames softly fixed.
    """
    bs, n, j, _ = rec_pose.shape
    dt = 1.0 / fps

    for b in range(bs):
        for t in range(1, n):
            delta = rec_pose[b, t, joint_idx] - rec_pose[b, t-1, joint_idx]
            # unwrap to avoid 360° jumps
            delta = (delta + 180) % 360 - 180
            speed = np.linalg.norm(delta) / dt  # deg/s

            if speed > abnormal_threshold_deg_per_s:
                rec_pose[b, t, joint_idx] = (
                    rec_pose[b, t-1, joint_idx] + delta * blend_factor
                )

    return rec_pose

def load_framewise_pose_file(path):
    """
    Load a text file whose each line is: joint*3 floats (no translation).
    Returns float32 array of shape (F, D).
    """
    import numpy as np
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append([float(x) for x in s.split()])
    return np.asarray(rows, dtype=np.float32)


class BaseTrainer(object):
    def __init__(self, args, sp, ap, tp):
        
        global inference_model, _vq_model_face, _vq_model_upper, _vq_model_hands, \
       _vq_model_lower, _global_motion, \
       _joint_mask_face, _joint_mask_upper, _joint_mask_hands, _joint_mask_lower, \
       _joints, _log_softmax, _test_loader

        
        hf_dir = "hf"
        # print(f"args.out_path: {args.out_path}")
        if not os.path.exists(args.out_path + "custom/" + hf_dir + "/"):
            os.makedirs(args.out_path + "custom/" + hf_dir + "/")
        # print(f"ap: {ap}")
        # line135
        sf.write(args.out_path + "custom/" + hf_dir + "/tmp.wav", ap[1], ap[0])
        self.audio_path = args.out_path + "custom/" + hf_dir + "/tmp.wav"
        audio, ssr = librosa.load(self.audio_path)
        ap = (ssr, audio)
        self.args = args
        self.rank = 0 # dist.get_rank()
       
        #self.checkpoint_path = args.out_path + "custom/" + args.name + args.notes + "/" #wandb.run.dir #args.cache_path+args.out_path+"/"+args.name
        self.checkpoint_path = args.out_path + "custom/" + hf_dir + "/" 
        if self.rank == 0:
            self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test", smplx_path=sp, audio_path=ap, text_path=tp)
            _test_loader = torch.utils.data.DataLoader(
                self.test_data, 
                batch_size=1,  
                shuffle=False,  
                num_workers=args.loader_workers,
                drop_last=False,
            )
        logger.info(f"Init test dataloader success")

        if _global_inference == False:
            
            model_module = __import__(f"models.{args.model}", fromlist=["something"])
            
            if args.ddp:
                inference_model = getattr(model_module, args.g_name)(args).to(device)
                process_group = torch.distributed.new_group()
                inference_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(inference_model, process_group)   
                inference_model = DDP(inference_model, device_ids=[self.rank], output_device=self.rank,
                                 broadcast_buffers=False, find_unused_parameters=False)
            else: 
                # inference_model = torch.nn.DataParallel(getattr(model_module, args.g_name)(args), args.gpus).to(device)
                device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
                inference_model = getattr(model_module, args.g_name)(args).to(device)

            
            if self.rank == 0:
                logger.info(f"init {args.g_name} success")

            self.args = args
            _joints = self.test_data.joints
            self.ori_joint_list = joints_list[self.args.ori_joints]
            self.tar_joint_list_face = joints_list["mixamo_face"]
            self.tar_joint_list_upper = joints_list["mixamo_upper"]
            self.tar_joint_list_hands = joints_list["mixamo_hand"]
            self.tar_joint_list_lower = joints_list["mixamo_lower"]
    
            _joints  = 52
           
            _joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            for joint_name in self.tar_joint_list_face:
                _joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]-3:self.ori_joint_list[joint_name][1]-3] = 1
                
            _joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            for joint_name in self.tar_joint_list_upper:
                _joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]-3:self.ori_joint_list[joint_name][1]-3] = 1
                
            _joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            for joint_name in self.tar_joint_list_hands:
                _joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]-3:self.ori_joint_list[joint_name][1]-3] = 1
                
            _joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            for joint_name in self.tar_joint_list_lower:
                if joint_name == 'Hips':
                    _joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]-3] = 1 
                else:
                    _joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]-3:self.ori_joint_list[joint_name][1]-3] = 1
    
            self.tracker = other_tools_hf.EpochTracker(["fid", "l1div", "bc", "rec", "trans", "vel", "transv", 'dis', 'gen', 'acc', 'transa', 'exp', 'lvd', 'mse', "cls", "rec_face", "latent", "cls_full", "cls_self", "cls_word", "latent_word","latent_self"], [False,True,True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False,False,False])
            
            vq_model_module = __import__(f"models.motion_representation", fromlist=["something"])
            self.args.vae_layer = 2
            self.args.vae_length = 256
            
            # face model 
            self.args.vae_test_dim = 51
            _vq_model_face = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(device)
            other_tools.load_checkpoints(_vq_model_face, pretrained_dir + "/face/face_785.bin", args.e_name)
            # Lab/mixamo_v2/pretrained/mixamo/face/last_699.bin
            # self.args.code_path + "/pretrained/mixamo/face/last_699.bin"
    
            # upper model
            self.args.vae_test_dim = 66
            _vq_model_upper = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(device)
            other_tools.load_checkpoints(_vq_model_upper,  pretrained_dir + "/upper/upper_1041.bin", args.e_name)
    
            # hand model
            self.args.vae_test_dim = 192
            _vq_model_hands = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(device)
            other_tools.load_checkpoints(_vq_model_hands,  pretrained_dir + "/hands/hands_795.bin", args.e_name)
    
            # lower model
            self.args.vae_test_dim = 59
            self.args.vae_layer = 4
            _vq_model_lower = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(device)
            other_tools.load_checkpoints(_vq_model_lower,  pretrained_dir + "/lowerfoot/lowerfoot_619.bin", args.e_name)
    
            # global motion
            self.args.vae_test_dim = 59
            self.args.vae_layer = 4
            _global_motion = getattr(vq_model_module, "VAEConvZero")(self.args).to(device)
            other_tools.load_checkpoints(_global_motion,  pretrained_dir + "/lowerfoot/lowerfoot_619.bin", args.e_name) # last_579
            
            self.args.vae_test_dim = 312
            self.args.vae_layer = 4
            self.args.vae_length = 240

            self.vae_codebook_size = 256
    
            _vq_model_face.eval()
            _vq_model_upper.eval()
            _vq_model_hands.eval()
            _vq_model_lower.eval()
            _global_motion.eval()

            _log_softmax = nn.LogSoftmax(dim=2).to(device)

    
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        # selection_array = torch.from_numpy(selection_array).to(device)
        selection_array = torch.from_numpy(selection_array.astype('float32')).to(device)
        original_shape_t = torch.zeros((n, 156)).to(device) # Số joint nhân 3
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t  
    
    def _load_data(self, dict_data):
        tar_pose_raw = dict_data["pose"]
        tar_pose = tar_pose_raw[:, :, :156].to(device)
        tar_contact = tar_pose_raw[:, :, 156:156+2].to(device)
        tar_contact = torch.zeros_like(tar_contact) # nl
        tar_trans = dict_data["trans"].to(device)
        tar_trans = torch.zeros_like(tar_trans) # nl
        tar_exps = dict_data["facial"].to(device) # [1, 1]
        in_audio = dict_data["audio"].to(device) 
        in_word = None
        tar_id = dict_data["id"].to(device).long()
        # *********
        tar_id = torch.where((tar_id >= 25) | (tar_id < 0), torch.tensor(0, device=tar_id.device), tar_id)
        # *********

        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], _joints

        tar_pose_face = tar_exps
        # print(f"tar_pose_face: {tar_pose_face}, {tar_pose_face.shape}")
        
        tar_pose_hands = tar_pose[:, :, _joint_mask_hands.astype(bool)]
        tar_pose_hands = rc.euler_angles_to_matrix(tar_pose_hands.reshape(bs, n, 32, 3), "YXZ")
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 32*6)

        tar_pose_upper = tar_pose[:, :, _joint_mask_upper.astype(bool)] # ([bs, 64, 42])
        tar_pose_upper = rc.euler_angles_to_matrix(tar_pose_upper.reshape(bs, n, 11, 3), "YXZ")
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 11*6)

        tar_pose_leg = tar_pose[:, :, _joint_mask_lower.astype(bool)]
        tar_pose_leg = torch.zeros_like(tar_pose_leg) # nl
        tar_pose_leg = rc.euler_angles_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3), "YXZ")
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)

        tar4dis = torch.cat([tar_pose_upper, tar_pose_hands, tar_pose_leg], dim=2)

        tar_index_value_face_top = _vq_model_face.map2index(tar_pose_face) # bs*n/4
        tar_index_value_upper_top = _vq_model_upper.map2index(tar_pose_upper) # bs*n/4
        tar_index_value_hands_top = _vq_model_hands.map2index(tar_pose_hands) # bs*n/4
        tar_index_value_lower_top = _vq_model_lower.map2index(tar_pose_lower) # bs*n/4
      
        latent_face_top = _vq_model_face.map2latent(tar_pose_face) # bs*n/4
        latent_upper_top = _vq_model_upper.map2latent(tar_pose_upper) # bs*n/4
        latent_hands_top = _vq_model_hands.map2latent(tar_pose_hands) # bs*n/4
        latent_lower_top = _vq_model_lower.map2latent(tar_pose_lower) # bs*n/4
        
        latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2)
        
        index_in = torch.stack([tar_index_value_upper_top, tar_index_value_hands_top, tar_index_value_lower_top], dim=-1).long()
        
        tar_pose_6d =rc.euler_angles_to_matrix(tar_pose.reshape(bs, n, 52, 3), "YXZ")
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 52*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)

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
    
    def _g_test(self, loaded_data):
        start4 = time.time()
        mode = 'test'
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], _joints 
        
        # print(f"bs, n, j: {bs}, {n}, {j}") # 1, 1950, 2


        tar_pose = loaded_data["tar_pose"]
        # tar_beta = loaded_data["tar_beta"]
        in_word =None# loaded_data["in_word"]
        tar_exps = loaded_data["tar_exps"]
        tar_contact = loaded_data["tar_contact"]
        tar_contact = torch.zeros_like(tar_contact) # nl
        in_audio = loaded_data["in_audio"]
        tar_trans = loaded_data["tar_trans"]
        tar_trans = torch.zeros_like(tar_trans) # nl

        # remain = n%8
        # if remain != 0:
        #     tar_pose = tar_pose[:, :-remain, :]
        #     # tar_beta = tar_beta[:, :-remain, :]
        #     tar_trans = tar_trans[:, :-remain, :]
        #     # in_word = in_word[:, :-remain]
        #     tar_exps = tar_exps[:, :-remain, :]
        #     tar_contact = tar_contact[:, :-remain, :]
        #     n = n - remain

         # face
        tar_pose_face = tar_exps

        # hands
        tar_pose_hands = tar_pose[:, :, _joint_mask_hands.astype(bool)]
        tar_pose_hands = rc.euler_angles_to_matrix(tar_pose_hands.reshape(bs, n, 32, 3), "YXZ")
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 32*6)

        # upper
        tar_pose_upper = tar_pose[:, :, _joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.euler_angles_to_matrix(tar_pose_upper.reshape(bs, n, 11, 3), "YXZ")
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 11*6)

        # lower
        tar_pose_leg = tar_pose[:, :, _joint_mask_lower.astype(bool)]
        tar_pose_leg = torch.zeros_like(tar_pose_leg) # nl
        tar_pose_leg = rc.euler_angles_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3), "YXZ")
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)
        
        tar_pose_6d = rc.euler_angles_to_matrix(tar_pose.reshape(bs, n, 52, 3), "YXZ")
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 52*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
 
        in_id_tmp = loaded_data['tar_id']

        min_len = min(latent_all.shape[1], in_id_tmp.shape[1])
        
        mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+2).float().to(device)
        mask_val[:, :self.args.pre_frames, :] = 0.0
        mask_val = mask_val[:, :min_len, :]
        
        latent_all_tmp = latent_all[:, :64, :]
        # latent_all_tmp = torch.zeros_like(latent_all_tmp)

        in_id_tmp = in_id_tmp[:, :64, :]
        # in_id_tmp = torch.zeros_like(in_id_tmp)

        end4 = time.time()
        # print(f"Thời gian chạy trước net_out_val: {end4 - start4:.3f} giây")

        # print(f"in_audio: {in_audio}")

        # print(f"in_audio: {in_audio.shape}")
        # print(f"latent_all_tmp: {latent_all_tmp.shape}")
        # print(f"mask_val: {mask_val.shape}")
        # print(f"in_id_tmp: {in_id_tmp.shape}")
        # print(f"tar_id: {loaded_data['tar_id'].shape}")


        # print(f"in_audio net_out_val: {in_audio}, {in_audio.shape}")
        start5 = time.time()
        net_out_val = inference_model(
            in_audio = in_audio,
            in_word=None, #in_word_tmp,
            mask=mask_val,
            in_motion = latent_all_tmp,
            in_id = in_id_tmp,
            use_attentions=True, # use_word=False
        )
        end5 = time.time()
        # print(f"Thời gian chạy net_out_val: {end5 - start5:.3f} giây")

        start6 = time.time()
        if self.args.cu != 0:
            rec_index_upper = _log_softmax(net_out_val["cls_upper"]).reshape(-1, self.args.vae_codebook_size)
            _, rec_index_upper = torch.max(rec_index_upper.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
            #rec_upper = _vq_model_upper.decode(rec_index_upper)
        else:
            _, rec_index_upper, _, _ = _vq_model_upper.quantizer(net_out_val["rec_upper"])
            #rec_upper = _vq_model_upper.decoder(rec_index_upper)
        if self.args.cl != 0:
            rec_index_lower = _log_softmax(net_out_val["cls_lower"]).reshape(-1, self.args.vae_codebook_size)
            _, rec_index_lower = torch.max(rec_index_lower.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
            #rec_lower = _vq_model_lower.decode(rec_index_lower)
        else:
            _, rec_index_lower, _, _ = _vq_model_lower.quantizer(net_out_val["rec_lower"])
            #rec_lower = _vq_model_lower.decoder(rec_index_lower)
        if self.args.ch != 0:
            rec_index_hands = _log_softmax(net_out_val["cls_hands"]).reshape(-1, self.args.vae_codebook_size)
            _, rec_index_hands = torch.max(rec_index_hands.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
            #rec_hands = _vq_model_hands.decode(rec_index_hands)
        else:
            _, rec_index_hands, _, _ = _vq_model_hands.quantizer(net_out_val["rec_hands"])
            #rec_hands = _vq_model_hands.decoder(rec_index_hands)
        if self.args.cf != 0:
            rec_index_face = _log_softmax(net_out_val["cls_face"]).reshape(-1, self.args.vae_codebook_size)
            _, rec_index_face = torch.max(rec_index_face.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
            #rec_face = _vq_model_face.decoder(rec_index_face)
        else:
            _, rec_index_face, _, _ = _vq_model_face.quantizer(net_out_val["rec_face"])
            #rec_face = _vq_model_face.decoder(rec_index_face)

        if self.args.cu != 0:
            rec_upper_last = _vq_model_upper.decode(rec_index_upper)
        else:
            rec_upper_last = _vq_model_upper.decoder(rec_index_upper)
        if self.args.cl != 0:
            rec_lower_last = _vq_model_lower.decode(rec_index_lower)
        else:
            rec_lower_last = _vq_model_lower.decoder(rec_index_lower)
        if self.args.ch != 0:
            rec_hands_last = _vq_model_hands.decode(rec_index_hands)
        else:
            rec_hands_last = _vq_model_hands.decoder(rec_index_hands)
        if self.args.cf != 0:
            rec_face_last = _vq_model_face.decode(rec_index_face)
        else:
            rec_face_last = _vq_model_face.decoder(rec_index_face)

        rec_exps = rec_face_last

        rec_lower_last = torch.zeros_like(rec_lower_last) # nl
        rec_pose_legs = rec_lower_last[:, :, :54]
        bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
        
        rec_pose_upper = rec_upper_last.reshape(bs, n, 11, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 11*3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, _joint_mask_upper, bs*n)
        
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = torch.zeros_like(rec_pose_lower) # nl
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, _joint_mask_lower, bs*n)
        rec_pose_lower_recover = torch.zeros_like(rec_pose_lower_recover)

        rec_pose_hands = rec_hands_last.reshape(bs, n, 32, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 32*3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, _joint_mask_hands, bs*n)
        
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
        rec_trans_v_s = rec_lower_last[:, :, 54:57]

        rec_trans = torch.zeros_like(rec_trans_v_s) # nl
        # print(f"rec_trans for: {rec_trans.shape}")
        latent_last = torch.cat([rec_pose, rec_trans, rec_lower_last[:, :, 57:59]], dim=-1)

        end6 = time.time()
        # print(f"Thời gian chạy sau net_out_val: {end6 - start6:.3f} giây")

        # print(f"net_out_vallllllll: {net_out_val}")
        return {
            'rec_pose': rec_pose,
            'rec_trans': rec_trans,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            # 'tar_beta': tar_beta,
            'tar_trans': tar_trans,
            'rec_exps': rec_exps,
        }

# --- Config ---
TARGET_SR = 16000
POSE_FPS = 30
POSE_LEN = 64
CHUNK_SAMPLES = int(round(TARGET_SR * (POSE_LEN / POSE_FPS)))  # ~34133

# --- Shared state ---
audio_buffer = np.empty((0,), dtype=np.float32)
buffer_lock = threading.Lock()
stop_event = threading.Event()



def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    if status:
        print(f"[Audio Warning] {status}")
    mono = indata[:, 0].astype(np.float32)

    with buffer_lock:
        audio_buffer = np.concatenate((audio_buffer, mono))

def chunk_consumer():
    global audio_buffer
    print(f"arget chunk size: {CHUNK_SAMPLES} samples (~{POSE_LEN / POSE_FPS:.2f}s)")
    
    smplx_path = None
    text_path = None
    args = config.parse_args()

    data, samplerate = sf.read("/Users/caocongdanh/Downloads/delete/1_fufu_0_1_1.wav")
    input_sample = (TARGET_SR, data)
    trainer = BaseTrainer(args, sp=smplx_path, ap=input_sample, tp=text_path)
    other_tools_hf.load_checkpoints(inference_model, test_checkpoint, args.g_name)
    batch_data = next(iter(_test_loader))

    start2 = time.time()
    loaded_data = trainer._load_data(dict_data=batch_data)
    # print(f"loaded_data: {loaded_data.keys()}")

    inference_model.eval()

    chunk_id = 0
    while not stop_event.is_set():
        with buffer_lock:
            if audio_buffer.shape[0] >= CHUNK_SAMPLES:
                chunk = audio_buffer[:CHUNK_SAMPLES]
                audio_buffer = audio_buffer[CHUNK_SAMPLES:]
            else:
                chunk = None
        t = time.time()

        # if chunk is not None:
        #     filename = f"/Users/caocongdanh/Study/NCU/Lab/source/audio/chunk_{chunk_id:05d}.wav"  # VD: chunk_00001.wav
        #     print(f"chunk length: {len(chunk)}")
        #     sf.write(filename, chunk, samplerate=16000, subtype="PCM_16")
        #     chunk_id += 1

        if chunk is not None:
            # print(f"[INFER] Processing chunk with shape: {chunk.shape}")
            in_audio_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
            audio_input = (TARGET_SR, in_audio_tensor)
            loaded_data['in_audio'] = audio_input[1]
            print(f"\n\nAudio chunk {chunk_id}: {audio_input[1]}")
            # print(f"in_audio before _g_test: {loaded_data['in_audio']}, {loaded_data['in_audio'].shape}")

            # inference
            net_out = trainer._g_test(loaded_data)

             # Load names
            with open(args.code_path + '/scripts/EMAGE_2024/utils/assets/names_only.json', 'r') as f:
                names_data = json.load(f)
            names = names_data['name']  # should be a list of 51 items

            bvh_path = args.code_path + "/scripts/EMAGE_2024/utils/assets/1_fufu_0_1_1_first64.bvh"  
            bvh_pose_master = load_framewise_pose_file(bvh_path)     # shape: (F_bvh, j*3)

            tar_pose = net_out['tar_pose']
            rec_pose = net_out['rec_pose']
            tar_exps = net_out['tar_exps']
            # tar_beta = net_out['tar_beta']
            rec_trans = net_out['rec_trans']
            tar_trans = net_out['tar_trans']
            rec_exps = net_out['rec_exps']
            # print(rec_pose.shape, tar_pose.shape)
            bs, n, j = tar_pose.shape[0], tar_pose.shape[1], _joints

            # print(f"N: {n}")

            rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*64, j, 6)) ## change 64
            rec_pose = rc.matrix_to_euler_angles(rec_pose, "YXZ").reshape(bs, 64, j, 3) 

            # print(f"rec_pose: {rec_pose}, {rec_pose.shape}")
            # rec_pose = np.rad2deg(rec_pose.cpu().numpy())
            rec_pose = np.rad2deg(rec_pose.detach().cpu().numpy())

            # rec_pose = fix_abnormal_joint_motion_soft(
            #     rec_pose,
            #     joint_idx=27,
            #     fps=30.0,
            #     abnormal_threshold_deg_per_s=600.0,
            #     blend_factor=0.2  # tweak to control how much smoothing
            # )
        
            rec_pose = rec_pose.reshape(bs*64, j*3)

            trans = torch.zeros_like(rec_trans)
            trans= rec_trans.reshape(bs*64, 3).cpu().numpy()

            bvh_pose = bvh_pose_master  # (F_bvh, j*3)

            if bvh_pose.shape[0] < rec_pose.shape[0]:
                repeats = -(-rec_pose.shape[0] // bvh_pose.shape[0])  # ceiling division
                bvh_pose = np.tile(bvh_pose, (repeats, 1))
            
            bvh_pose = bvh_pose[:rec_pose.shape[0], :]
            use_frames = rec_pose.shape[0]  
            
            trans[:use_frames, :] = bvh_pose[:use_frames, 0:3]
            
            rec_pose[:use_frames, -24:] = bvh_pose[:use_frames, -24:]
            
            rec_pose[:use_frames, 3:6] = bvh_pose[:use_frames, 3:6]
            
            rec_pose = np.concatenate([trans, rec_pose], axis=1)  # (frames, 3 + j*3)
            
            total_length = 64

            # Saving body motion (bvh): only motion, not hirarchy
            results_save_path = "/Users/caocongdanh/Study/NCU/Lab/source/outputs/"

            np.savetxt(
                f"{results_save_path}res_motion_{chunk_id}.bvh",
                rec_pose,
                fmt='%.6f',
                delimiter=' '
            )

            # print(f"final rec_pose: {rec_pose.shape}")

            # bvh full
            # get hirarchy
            file_content_length = 316
            with open("/Users/caocongdanh/Study/NCU/Lab/source/EMAGEMixamo/scripts/EMAGE_2024/utils/assets/1_fufu_0_7_7_full.bvh", 'r') as f:
                tmpl_lines = f.readlines()
            header_lines = tmpl_lines[:file_content_length]

            offset_line  = tmpl_lines[file_content_length]
            offset_data  = np.fromstring(offset_line, dtype=float, sep=' ')

            header_lines[file_content_length - 2] = f"Frames: {rec_pose.shape[0]}\n"

            # Saving bvh full
            with open(f"{results_save_path}res_full_{chunk_id}.bvh", 'w') as f:
                f.writelines(header_lines)
                np.savetxt(f, rec_pose, fmt="%.6f", delimiter=" ")

            # Saving audio
            in_audio = loaded_data["in_audio"]
            audio_np = in_audio.squeeze().cpu().numpy()
            audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-8)  # normalize to [-1, 1]
            audio_int16 = (audio_np * 32767).astype(np.int16)
            sample_rate = getattr(args, "audio_sample_rate", 16000)
            wavfile.write(f"{results_save_path}audio_{chunk_id}.wav", sample_rate, audio_int16)

            ## Save face
            # rec_exps = rec_exps.cpu().numpy().reshape(bs*64, 51)
            rec_exps = rec_exps.detach().cpu().numpy().reshape(bs*64, 51)
            rec_exps = np.clip(rec_exps, 0, 1) # Clip the values to range of 0-1
            frames = [{"weights": list(map(float, frame))} for frame in rec_exps]

            # Save lại audio
            save_dict = {
                "name": names,
                "frames": frames
            }

            save_path = f"{results_save_path}face_{chunk_id}.json"
            with open(save_path, 'w') as f:
                json.dump(save_dict, f, indent=2)

            # test_demo = chunk_id
            # data_tools.result2target_vis("mixamo_joint_full", results_save_path, results_save_path, test_demo, mode="all_or_lower", verbose=False)

            print(f"inference time: {time.time()-t}")
            chunk_id += 1
        else:
            time.sleep(0.01)

        

def main():
    try:
        print("Starting audio input stream...")
        stream = sd.InputStream(
            channels=1,
            samplerate=TARGET_SR,
            callback=audio_callback,
            blocksize=1024,
            dtype='float32'
        )

        # Start audio stream explicitly
        stream.start()

        # Start consumer thread (daemon so it won't block exit)
        
        consumer_thread = threading.Thread(target=chunk_consumer, daemon=True)
        consumer_thread.start()
        

        print("Microphone is recording... Press Ctrl+C to stop.")
        while not stop_event.is_set():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Stopping...")
        stop_event.set()

    finally:
        print("Cleaning up...")
        try:
            stream.stop()
            stream.close()
        except Exception as e:
            print(f"Error closing stream: {e}")
        time.sleep(0.5)

if __name__ == "__main__":
    main()
