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
import sounddevice as sd


_global_inference = False
inference_model = None

_vq_model_face = None
_vq_model_upper = None
_vq_model_hands = None
_vq_model_lower = None
_global_motion = None

# _joint_mask_face = None
# _joint_mask_upper = None
# _joint_mask_hands = None 
# _joint_mask_lower = None

current_dir = os.path.dirname(os.path.abspath(__file__))
pretrained_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "pretrained/mixamo"))

test_checkpoint = pretrained_dir + "/last_92.bin"

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
        global inference_model
        global _vq_model_face
        global _vq_model_upper
        global _vq_model_hands
        global _vq_model_lower
        global _global_motion 

        # global _joint_mask_face
        # global _joint_mask_upper
        # global _joint_mask_hands
        # global _joint_mask_lower
        
        hf_dir = "hf"
        # print(f"args.out_path: {args.out_path}")
        if not os.path.exists(args.out_path + "custom/" + hf_dir + "/"):
            os.makedirs(args.out_path + "custom/" + hf_dir + "/")
        sf.write(args.out_path + "custom/" + hf_dir + "/tmp.wav", ap[1], ap[0])
        self.audio_path = args.out_path + "custom/" + hf_dir + "/tmp.wav"
        audio, ssr = librosa.load(self.audio_path)
        ap = (ssr, audio)
        # self.args = args
        self.rank = 0 # dist.get_rank()
       
        #self.checkpoint_path = args.out_path + "custom/" + args.name + args.notes + "/" #wandb.run.dir #args.cache_path+args.out_path+"/"+args.name
        self.checkpoint_path = args.out_path + "custom/" + hf_dir + "/" 
        if self.rank == 0:
            self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test", smplx_path=sp, audio_path=ap, text_path=tp)
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
            inference_model = getattr(model_module, args.g_name)(args).to(self.rank)
            process_group = torch.distributed.new_group()
            inference_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(inference_model, process_group)   
            inference_model = DDP(inference_model, device_ids=[self.rank], output_device=self.rank,
                             broadcast_buffers=False, find_unused_parameters=False)
        else: 
            inference_model = torch.nn.DataParallel(getattr(model_module, args.g_name)(args), args.gpus).cuda()
        
        if self.rank == 0:
            # logger.info(inference_model)
            logger.info(f"init {args.g_name} success")
    
            # self.smplx = smplx.create(
            # self.args.data_path_1+"smplx_models/", 
            #     model_type='smplx',
            #     gender='NEUTRAL_2020', 
            #     use_face_contour=False,
            #     num_betas=300,
            #     num_expression_coeffs=100, 
            #     ext='npz',
            #     use_pca=False,
            # ).to(self.rank).eval()
        self.args = args
        self.joints = self.test_data.joints
        # self.ori_joint_list = joints_list[self.args.ori_joints]
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["mixamo_face"]
        self.tar_joint_list_upper = joints_list["mixamo_upper"]
        self.tar_joint_list_hands = joints_list["mixamo_hand"]
        self.tar_joint_list_lower = joints_list["mixamo_lower"]

        # print(f"self.args.ori_joints: {self.args.ori_joints}")
        # print(f"self.ori_joint_list: {self.ori_joint_list}")

        self.joints = 52
       
        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
            
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
            
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
            
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        self.tracker = other_tools_hf.EpochTracker(["fid", "l1div", "bc", "rec", "trans", "vel", "transv", 'dis', 'gen', 'acc', 'transa', 'exp', 'lvd', 'mse', "cls", "rec_face", "latent", "cls_full", "cls_self", "cls_word", "latent_word","latent_self"], [False,True,True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False,False,False])

        if _global_inference == False:
            
            vq_model_module = __import__(f"models.motion_representation", fromlist=["something"])
            self.args.vae_layer = 2
            self.args.vae_length = 256
            
            # face model
            self.args.vae_test_dim = 51
            _vq_model_face = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
            other_tools.load_checkpoints(_vq_model_face, pretrained_dir + "/face/face_785.bin", args.e_name)
            # Lab/mixamo_v2/pretrained/mixamo/face/last_699.bin
            # self.args.code_path + "/pretrained/mixamo/face/last_699.bin"
    
            # upper model
            self.args.vae_test_dim = 66
            _vq_model_upper = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
            other_tools.load_checkpoints(_vq_model_upper,  pretrained_dir + "/upper/upper_1041.bin", args.e_name)
    
            # hand model
            self.args.vae_test_dim = 192
            _vq_model_hands = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
            other_tools.load_checkpoints(_vq_model_hands,  pretrained_dir + "/hands/hands_795.bin", args.e_name)
    
            # lower model
            self.args.vae_test_dim = 59
            self.args.vae_layer = 4
            _vq_model_lower = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
            other_tools.load_checkpoints(_vq_model_lower,  pretrained_dir + "/lowerfoot/lowerfoot_619.bin", args.e_name)
    
            # global motion
            self.args.vae_test_dim = 59
            self.args.vae_layer = 4
            _global_motion = getattr(vq_model_module, "VAEConvZero")(self.args).to(self.rank)
            other_tools.load_checkpoints(_global_motion,  pretrained_dir + "/lowerfoot/lowerfoot_619.bin", args.e_name)
            
            self.args.vae_test_dim = 312
            self.args.vae_layer = 4
            self.args.vae_length = 240

            self.vae_codebook_size = 256
    
            _vq_model_face.eval()
            _vq_model_upper.eval()
            _vq_model_hands.eval()
            _vq_model_lower.eval()
            _global_motion.eval()

        # self.cls_loss = nn.NLLLoss().to(self.rank)
        # self.reclatent_loss = nn.MSELoss().to(self.rank)
        # self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        # self.rec_loss = get_loss_func("GeodesicLoss").to(self.rank) 
        self.log_softmax = nn.LogSoftmax(dim=2).to(self.rank)

         # *********
        # self.loss_meters = {
        #     'val_all': other_tools.AverageMeter('val_all'),
        #     'rec_val': other_tools.AverageMeter('rec_val'),
        #     'vel_val': other_tools.AverageMeter('vel_val'),
        #     'acceleration_val': other_tools.AverageMeter('acceleration_val'),
            
        # }
        # self.best_epochs = {
        #     'val_all': [np.inf, 0],
        #     'rec_val': [np.inf, 0],
        #     'vel_val': [np.inf, 0],
        #     'acceleration_val': [np.inf, 0],
        # }
        # *********

    
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 156)).cuda() # Số joint nhân 3
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t  
    
    def _load_data(self, dict_data):
        tar_pose_raw = dict_data["pose"]
        # print(f"tar_pose_raw: {tar_pose_raw.shape}")
        # print(f"tar_pose_raw: {tar_pose_raw}, {tar_pose_raw.shape}")
        tar_pose = tar_pose_raw[:, :, :156].to(self.rank)
        # print(f"tar_pose: {tar_pose.shape}")
        # print(f"self.joint_mask_hands: len(self.joint_mask_hands)")
        tar_contact = tar_pose_raw[:, :, 156:156+2].to(self.rank)
        tar_contact = torch.zeros_like(tar_contact) # nl
        tar_trans = dict_data["trans"].to(self.rank)
        tar_trans = torch.zeros_like(tar_trans) # nl
        tar_exps = dict_data["facial"].to(self.rank) # [1, 1]
        # print(f"tar_exps: {tar_exps}, {tar_exps.shape}")
        in_audio = dict_data["audio"].to(self.rank) 
        in_word = None
        tar_id = dict_data["id"].to(self.rank).long()
        # *********
        tar_id = torch.where((tar_id >= 25) | (tar_id < 0), torch.tensor(0, device=tar_id.device), tar_id)
        # *********

        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

        tar_pose_face = tar_exps
        # print(f"tar_pose_face: {tar_pose_face}, {tar_pose_face.shape}")
        
        tar_pose_hands = tar_pose[:, :, self.joint_mask_hands.astype(bool)]
        tar_pose_hands = rc.euler_angles_to_matrix(tar_pose_hands.reshape(bs, n, 32, 3), "YXZ")
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 32*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)] # ([bs, 64, 42])
        tar_pose_upper = rc.euler_angles_to_matrix(tar_pose_upper.reshape(bs, n, 11, 3), "YXZ")
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 11*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
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
        mode = 'test'
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        tar_pose = loaded_data["tar_pose"]
        # tar_beta = loaded_data["tar_beta"]
        in_word =None# loaded_data["in_word"]
        tar_exps = loaded_data["tar_exps"]
        tar_contact = loaded_data["tar_contact"]
        tar_contact = torch.zeros_like(tar_contact) # nl
        in_audio = loaded_data["in_audio"]
        tar_trans = loaded_data["tar_trans"]
        tar_trans = torch.zeros_like(tar_trans) # nl


        remain = n%8
        if remain != 0:
            tar_pose = tar_pose[:, :-remain, :]
            # tar_beta = tar_beta[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            # in_word = in_word[:, :-remain]
            tar_exps = tar_exps[:, :-remain, :]
            tar_contact = tar_contact[:, :-remain, :]
            n = n - remain

         # face
        tar_pose_face = tar_exps

        # hands
        tar_pose_hands = tar_pose[:, :, self.joint_mask_hands.astype(bool)]
        tar_pose_hands = rc.euler_angles_to_matrix(tar_pose_hands.reshape(bs, n, 32, 3), "YXZ")
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 32*6)

        # upper
        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.euler_angles_to_matrix(tar_pose_upper.reshape(bs, n, 11, 3), "YXZ")
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 11*6)

        # lower
        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = torch.zeros_like(tar_pose_leg) # nl
        tar_pose_leg = rc.euler_angles_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3), "YXZ")
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)
        
        tar_pose_6d = rc.euler_angles_to_matrix(tar_pose.reshape(bs, n, 52, 3), "YXZ")
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 52*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
        
        rec_index_all_face = []
        rec_index_all_upper = []
        rec_index_all_lower = []
        rec_index_all_hands = []
        
        roundt = (n - self.args.pre_frames) // (self.args.pose_length - self.args.pre_frames)
        remain = (n - self.args.pre_frames) % (self.args.pose_length - self.args.pre_frames)
        round_l = self.args.pose_length - self.args.pre_frames

        for i in range(0, roundt):
            # in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            # audio fps is 16000 and pose fps is 30
            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames]
            in_id_tmp = loaded_data['tar_id'][:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+2).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0
            if i == 0:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]
            else:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]
                # print(latent_all_tmp.shape, latent_last.shape)
                latent_all_tmp[:, :self.args.pre_frames, :] = latent_last[:, -self.args.pre_frames:, :]
            
            net_out_val = inference_model(
                in_audio = in_audio_tmp,
                in_word=None, #in_word_tmp,
                mask=mask_val,
                in_motion = latent_all_tmp,
                in_id = in_id_tmp,
                use_attentions=True,
                use_word=False
            )
            
            if self.args.cu != 0:
                rec_index_upper = self.log_softmax(net_out_val["cls_upper"]).reshape(-1, self.args.vae_codebook_size)
                _, rec_index_upper = torch.max(rec_index_upper.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                #rec_upper = _vq_model_upper.decode(rec_index_upper)
            else:
                _, rec_index_upper, _, _ = _vq_model_upper.quantizer(net_out_val["rec_upper"])
                #rec_upper = _vq_model_upper.decoder(rec_index_upper)
            if self.args.cl != 0:
                rec_index_lower = self.log_softmax(net_out_val["cls_lower"]).reshape(-1, self.args.vae_codebook_size)
                _, rec_index_lower = torch.max(rec_index_lower.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                #rec_lower = _vq_model_lower.decode(rec_index_lower)
            else:
                _, rec_index_lower, _, _ = _vq_model_lower.quantizer(net_out_val["rec_lower"])
                #rec_lower = _vq_model_lower.decoder(rec_index_lower)
            if self.args.ch != 0:
                rec_index_hands = self.log_softmax(net_out_val["cls_hands"]).reshape(-1, self.args.vae_codebook_size)
                _, rec_index_hands = torch.max(rec_index_hands.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                #rec_hands = _vq_model_hands.decode(rec_index_hands)
            else:
                _, rec_index_hands, _, _ = _vq_model_hands.quantizer(net_out_val["rec_hands"])
                #rec_hands = _vq_model_hands.decoder(rec_index_hands)
            if self.args.cf != 0:
                rec_index_face = self.log_softmax(net_out_val["cls_face"]).reshape(-1, self.args.vae_codebook_size)
                _, rec_index_face = torch.max(rec_index_face.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
                #rec_face = _vq_model_face.decoder(rec_index_face)
            else:
                _, rec_index_face, _, _ = _vq_model_face.quantizer(net_out_val["rec_face"])
                #rec_face = _vq_model_face.decoder(rec_index_face)

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
            # if self.args.cf != 0:
            #     rec_face_last = _vq_model_face.decode(rec_index_face)
            # else:
            #     rec_face_last = _vq_model_face.decoder(rec_index_face)

            
            rec_lower_last = torch.zeros_like(rec_lower_last) # nl
            rec_pose_legs = rec_lower_last[:, :, :54]
            bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
            
            rec_pose_upper = rec_upper_last.reshape(bs, n, 11, 6)
            rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
            rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 11*3)
            rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
            
            rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
            rec_pose_lower = torch.zeros_like(rec_pose_lower) # nl
            rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
            rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
            rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
            
            rec_pose_hands = rec_hands_last.reshape(bs, n, 32, 6)
            rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
            rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 32*3)
            rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
            
            rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
            rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, j, 3))
            rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
            rec_trans_v_s = rec_lower_last[:, :, 54:57]
            # rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 1:2], 1/self.args.pose_fps, tar_trans[:, 0, 1:2])
            # rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
            # rec_y_trans = rec_trans_v_s[:,:,0:1]
            # rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
            rec_trans = torch.zeros_like(rec_trans_v_s) # nl
            latent_last = torch.cat([rec_pose, rec_trans, rec_lower_last[:, :, 57:59]], dim=-1)

        rec_index_face = torch.cat(rec_index_all_face, dim=1)
        rec_index_upper = torch.cat(rec_index_all_upper, dim=1)
        rec_index_lower = torch.cat(rec_index_all_lower, dim=1)
        rec_index_hands = torch.cat(rec_index_all_hands, dim=1)
        
        if self.args.cu != 0:
            rec_upper = _vq_model_upper.decode(rec_index_upper)
        else:
            rec_upper = _vq_model_upper.decoder(rec_index_upper)
        if self.args.cl != 0:
            rec_lower = _vq_model_lower.decode(rec_index_lower)
        else:
            rec_lower = _vq_model_lower.decoder(rec_index_lower)
        if self.args.ch != 0:
            rec_hands = _vq_model_hands.decode(rec_index_hands)
        else:
            rec_hands = _vq_model_hands.decoder(rec_index_hands)
        if self.args.cf != 0:
            rec_face = _vq_model_face.decode(rec_index_face)
        else:
            rec_face = _vq_model_face.decoder(rec_index_face)

        rec_exps = rec_face
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
        
        rec_pose_upper = rec_upper.reshape(bs, n, 11, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 11*3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
        
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = torch.zeros_like(rec_pose_lower) # nl
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = rc.matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9*6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
        rec_pose_lower_recover = torch.zeros_like(rec_pose_lower_recover)
        
        rec_pose_hands = rec_hands.reshape(bs, n, 32, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 32*3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
        
        # rec_pose_jaw = rec_pose_jaw.reshape(bs*n, 6)
        # rec_pose_jaw = rc.rotation_6d_to_matrix(rec_pose_jaw)
        # rec_pose_jaw = rc.matrix_to_axis_angle(rec_pose_jaw).reshape(bs*n, 1*3)
        # rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
        # rec_pose[:, 66:69] = rec_pose_jaw
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 

        to_global = rec_lower
        to_global[:, :, 54:57] = 0.0
        to_global[:, :, :54] = rec_lower2global
        rec_global = _global_motion(to_global)

        rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
        # rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 1:2], 1/self.args.pose_fps, tar_trans[:, 0, 1:2])
        # rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
        # rec_y_trans = rec_trans_v_s[:,:,0:1]
        # rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
        rec_trans = torch.zeros_like(rec_trans_v_s)
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


    def test_demo(self, epoch, ap):

        # Load names
        with open(self.args.code_path + '/scripts/EMAGE_2024/utils/assets/names_only.json', 'r') as f:
            names_data = json.load(f)
        names = names_data['name']  # should be a list of 51 items

        bvh_path = self.args.code_path + "/scripts/EMAGE_2024/utils/assets/1_fufu_0_1_1_first64.bvh"  
        bvh_pose_master = load_framewise_pose_file(bvh_path)     # shape: (F_bvh, j*3)

        
        filename = os.path.splitext(os.path.basename(ap))[0]
        
        results_save_path = self.checkpoint_path + f"{epoch}/"
        if os.path.exists(results_save_path): 
            import shutil
            shutil.rmtree(results_save_path)
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        
        # test_demo = "/data/nas07/PersonalData/danh/EMAGEMixamo/mixamo_english_30_4_811_all/test_norm/bvh_full/"  
        test_demo = "inference"
        # test_seq_list = os.listdir(test_demo)
        # test_seq_list.sort()

        # test_seq_list = self.test_data.selected_file
        
        align = 0 
        latent_out = []
        latent_ori = []
        l2_all = 0 
        lvel = 0
        inference_model.eval()
        # self.smplx.eval()
        # self.eval_copy.eval()
        result_path = ""
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                # print(its, "abc\n\n\n\n")
                loaded_data = self._load_data(batch_data)
                net_out = self._g_test(loaded_data)
                in_audio = loaded_data['in_audio']
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                # tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                # print(rec_pose.shape, tar_pose.shape)
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

                # interpolate to 30fps  
                # if (30/self.args.pose_fps) != 1:
                #     assert 30%self.args.pose_fps == 0
                #     n *= int(30/self.args.pose_fps)
                #     tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                #     rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                
                # print(rec_pose.shape, tar_pose.shape)
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                # rec_pose = rc.matrix_to_euler_angles(rec_pose, "YXZ").reshape(bs*n, j*3)
                rec_pose = rc.matrix_to_euler_angles(rec_pose, "YXZ").reshape(bs, n, j, 3) 

                rec_pose = np.rad2deg(rec_pose.cpu().numpy())



                ###------------Fix abnormal motion-----------------##
                # rec_pose is (bs, n, j, 3) in degrees
                rec_pose = fix_abnormal_joint_motion_soft(
                    rec_pose,
                    joint_idx=27, ## Joint thứ 28(RArm1 là joint 27 trong 0-base
                    fps=30.0,
                    abnormal_threshold_deg_per_s=600.0,
                    blend_factor=0.2  # tweak to control how much smoothing
                )
                ###------------Fix abnormal motion-----------------##
                
                rec_pose = rec_pose.reshape(bs*n, j*3)
                trans = torch.zeros_like(rec_trans)
                trans= rec_trans.reshape(bs*n, 3).cpu().numpy()

                
                # --- Make a working copy of BVH pose and repeat/trim to match current length ---
                bvh_pose = bvh_pose_master  # (F_bvh, j*3)

                
                if bvh_pose.shape[0] < rec_pose.shape[0]:
                    # Repeat rows (tile) until we have at least as many frames as rec_pose
                    repeats = -(-rec_pose.shape[0] // bvh_pose.shape[0])  # ceiling division
                    bvh_pose = np.tile(bvh_pose, (repeats, 1))
                
                # Trim to exact length
                bvh_pose = bvh_pose[:rec_pose.shape[0], :]
                use_frames = rec_pose.shape[0]  
                
                # --- 1) rec_trans <- first 3 values of the BVH per frame ---
                trans[:use_frames, :] = bvh_pose[:use_frames, 0:3]
                
                # --- 2) Last 8 joints of rec_pose <- BVH's last 8*3 values (last 24 channels) ---
                rec_pose[:use_frames, -24:] = bvh_pose[:use_frames, -24:]
                
                # --- 3) rec_pose[:, 3:6] <- BVH[:, 3:6] ---
                rec_pose[:use_frames, 3:6] = bvh_pose[:use_frames, 3:6]
                
                # Concatenate translation + pose channels for saving
                trans = trans
                rec_pose = np.concatenate([trans, rec_pose], axis=1)  # (frames, 3 + j*3)



                total_length += n
                
                
                # seq_name = test_seq_list[its].split('.')[0]  # safer filename
                result_path = f"{results_save_path}res_{filename}.bvh"
                with open(f"{results_save_path}result_raw_{filename}.bvh", 'w+') as f_real:
                    for line_id in range(rec_pose.shape[0]): #,args.pre_frames, args.pose_length
                        line_data = np.array2string(rec_pose[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                        f_real.write(line_data[1:-2]+'\n')

                # === Save input audio as .wav ===
                audio_np = in_audio.squeeze().cpu().numpy()
                audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-8)  # normalize to [-1, 1]
                audio_int16 = (audio_np * 32767).astype(np.int16)
                sample_rate = getattr(self.args, "audio_sample_rate", 16000)
                wavfile.write(f"{results_save_path}in_audio_{filename}.wav", sample_rate, audio_int16)


                ## Save face
                rec_exps = rec_exps.cpu().numpy().reshape(bs*n, 51)
                rec_exps = np.clip(rec_exps, 0, 1) # Clip the values to range of 0-1
                frames = [{"weights": list(map(float, frame))} for frame in rec_exps]
            
                save_dict = {
                    "name": names,
                    "frames": frames
                }
            
                save_path = f"{results_save_path}face_{filename}.json"
                with open(save_path, 'w') as f:
                    json.dump(save_dict, f, indent=2)
                
                        
                del rec_pose, tar_pose, rec_trans, net_out, loaded_data, in_audio
                torch.cuda.empty_cache()

            data_tools.result2target_vis("mixamo_joint_full", results_save_path, results_save_path, test_demo, mode="all_or_lower", verbose=False)
        # result = gr.Video(value=render_vid_path, visible=True)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")
        return result_path

@logger.catch
def emage(audio_path):
    smplx_path = None
    text_path = None
    rank = 0
    world_size = 1
    args = config.parse_args()

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    other_tools_hf.set_random_seed(args)
    other_tools_hf.print_exp_info(args)

    data, samplerate = sf.read(audio_path)
    audio_input = (samplerate, data)

    trainer = BaseTrainer(args, sp=smplx_path, ap=audio_input, tp=text_path)
    if _global_inference == False:
        other_tools_hf.load_checkpoints(inference_model, test_checkpoint, args.g_name)
    result = trainer.test_demo(999, ap=audio_path)
    return result

            
def record_audio(duration=5, samplerate=16000):
    print(f"🎙  Recording... {duration} s...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # chờ ghi âm xong
    print("Done")
    return samplerate, audio_data


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = '127.0.0.1'
    os.environ["MASTER_PORT"] = '8675'

    while True:
        try:
            cmd = input("Enter the number of second, (enter '0' to exit): ").strip()
            if cmd == '0':
                break

            duration = int(cmd)
            audio_input = record_audio(duration=duration, samplerate=16000)

            output_file = emage(audio_input)
            print(f"\nFinished. Result path: {output_file}\n")
            _global_inference = True
        except Exception as e:
            print(f"Error: {e}")


# if __name__ == "__main__":
#     audio_path = input("Enter audio path (.wav): ").strip()

#     if not os.path.isfile(audio_path):
#         print(f"Error'{audio_path}' no exist")
#         sys.exit(1)

#     os.environ["MASTER_ADDR"] = '127.0.0.1'
#     os.environ["MASTER_PORT"] = '8675'

#     output_file = emage(audio_path)
#     print(f"\nFinished. Result path: {output_file}\n")
