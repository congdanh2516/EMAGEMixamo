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

# micro
import sounddevice as sd
import queue

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
       _joints, _log_softmax

        
        hf_dir = "hf"
        # print(f"args.out_path: {args.out_path}")
        if not os.path.exists(args.out_path + "custom/" + hf_dir + "/"):
            os.makedirs(args.out_path + "custom/" + hf_dir + "/")
        # sf.write(args.out_path + "custom/" + hf_dir + "/tmp.wav", ap[1], ap[0])
        print(f"ap: {ap}")
        audio_data = ap[1]
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().cpu().numpy()
        audio_np = audio_data.squeeze()
        print(f"audio_data: {audio_data}, {audio_np.shape}")
        print(f"ap[0]: {ap[0]}")
        sf.write(
            args.out_path + "custom/" + hf_dir + "/tmp.wav",
            audio_np,
            ap[0]  # sample rate
        )


        self.audio_path = args.out_path + "custom/" + hf_dir + "/tmp.wav"
        audio, ssr = librosa.load(self.audio_path)
        ap = (ssr, audio)
        self.args = args
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
            # _joints = self.test_data.joints
            self.ori_joint_list = joints_list[self.args.ori_joints]
            self.tar_joint_list_face = joints_list["mixamo_face"]
            self.tar_joint_list_upper = joints_list["mixamo_upper"]
            self.tar_joint_list_hands = joints_list["mixamo_hand"]
            self.tar_joint_list_lower = joints_list["mixamo_lower"]
    
            _joints  = 52
           
            _joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            for joint_name in self.tar_joint_list_face:
                _joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
                
            _joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            for joint_name in self.tar_joint_list_upper:
                _joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
                
            _joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            for joint_name in self.tar_joint_list_hands:
                _joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
                
            _joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            for joint_name in self.tar_joint_list_lower:
                _joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
    
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
        # print(f"tar_exps: {tar_exps}, {tar_exps.shape}")
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
        mode = 'test'
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], _joints 
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
        
        rec_index_all_face = []
        rec_index_all_upper = []
        rec_index_all_lower = []
        rec_index_all_hands = []
        
        roundt = (n - self.args.pre_frames) // (self.args.pose_length - self.args.pre_frames)
        remain = (n - self.args.pre_frames) % (self.args.pose_length - self.args.pre_frames)
        round_l = self.args.pose_length - self.args.pre_frames

        print(f"roundt: {roundt}")
        roundt = 1

        for i in range(0, roundt):
            # in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            # audio fps is 16000 and pose fps is 30
            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames]
            # in_audio_tmp = in_audio
            in_id_tmp = loaded_data['tar_id'][:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+2).float().to(device)
            mask_val[:, :self.args.pre_frames, :] = 0.0
            if i == 0:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]
            else:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]
                # print(latent_all_tmp.shape, latent_last.shape)
                latent_all_tmp[:, :self.args.pre_frames, :] = latent_last[:, -self.args.pre_frames:, :]

            print(f"in_audio: {in_audio.shape}")
            print(f"latent_all_tmp: {latent_all.shape}")
            print(f"mask_val: {mask_val.shape}")
            print(f"in_id_tmp: {in_id_tmp.shape}")
            
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
            rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, _joint_mask_upper, bs*n)
            
            rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
            rec_pose_lower = torch.zeros_like(rec_pose_lower) # nl
            rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
            rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
            rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, _joint_mask_lower, bs*n)
            
            rec_pose_hands = rec_hands_last.reshape(bs, n, 32, 6)
            rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
            rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 32*3)
            rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, _joint_mask_hands, bs*n)
            
            rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
            rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, j, 3))
            rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
            rec_trans_v_s = rec_lower_last[:, :, 54:57]
            # rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 1:2], 1/self.args.pose_fps, tar_trans[:, 0, 1:2])
            # rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
            # rec_y_trans = rec_trans_v_s[:,:,0:1]
            # rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
            rec_trans = torch.zeros_like(rec_trans_v_s) # nl
            print(f"rec_trans for: {rec_trans.shape}")
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
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, _joint_mask_upper, bs*n)
        
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = torch.zeros_like(rec_pose_lower) # nl
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = rc.matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9*6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, _joint_mask_lower, bs*n)
        rec_pose_lower_recover = torch.zeros_like(rec_pose_lower_recover)
        
        rec_pose_hands = rec_hands.reshape(bs, n, 32, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 32*3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, _joint_mask_hands, bs*n)
        

        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 

        to_global = rec_lower
        to_global[:, :, 54:57] = 0.0
        to_global[:, :, :54] = rec_lower2global
        rec_global = _global_motion(to_global)

        rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
        rec_trans = torch.zeros_like(rec_trans_v_s)
        print(f"rec_trans: {rec_trans.shape}")
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



    def _g_test_micro(self, loaded_data):
        start4 = time.time()
        mode = 'test'
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], _joints 
        
        print(f"bs, n, j: {bs}, {n}, {j}") # 1, 1950, 2


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


        # Lấy toàn bộ id
        in_id_tmp = loaded_data['tar_id']  

        # Tạo mask cho toàn bộ sequence
        mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+2).float().to(device)
        mask_val[:, :self.args.pre_frames, :] = 0.0

        # Lấy toàn bộ latent (không cần chắp nối từng chunk)
        latent_all_tmp = latent_all

        # in_id_tmp = loaded_data['tar_id']
        # in_id_tmp = loaded_data['tar_id'][:, 0*(round_l):(0+1)*(round_l)+self.args.pre_frames]
        in_id_tmp = loaded_data['tar_id']
        min_len = min(latent_all.shape[1], in_id_tmp.shape[1])
        latent_all_tmp = latent_all[:, :64, :]
        mask_val = mask_val[:, :min_len, :]
        in_id_tmp = in_id_tmp[:, :64, :]

        latent_all_tmp = torch.zeros_like(latent_all_tmp)
        in_id_tmp = torch.zeros_like(in_id_tmp)

        end4 = time.time()
        print(f"Thời gian chạy trước net_out_val: {end4 - start4:.3f} giây")

        print(f"in_audio: {in_audio}")

        print(f"in_audio: {in_audio.shape}")
        print(f"latent_all_tmp: {latent_all_tmp.shape}")
        print(f"mask_val: {mask_val.shape}")
        print(f"in_id_tmp: {in_id_tmp.shape}")
        print(f"tar_id: {loaded_data['tar_id'].shape}")


        start5 = time.time()
        net_out_val = inference_model(
            in_audio = in_audio,
            in_word=None, #in_word_tmp,
            mask=mask_val,
            in_motion = latent_all_tmp,
            in_id = in_id_tmp,
            use_attentions=True,
            use_word=False
        )
        end5 = time.time()
        print(f"Thời gian chạy net_out_val: {end5 - start5:.3f} giây")

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
        
        rec_pose_hands = rec_hands_last.reshape(bs, n, 32, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 32*3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, _joint_mask_hands, bs*n)
        
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
        rec_trans_v_s = rec_lower_last[:, :, 54:57]

        rec_trans = torch.zeros_like(rec_trans_v_s) # nl
        print(f"rec_trans for: {rec_trans.shape}")
        latent_last = torch.cat([rec_pose, rec_trans, rec_lower_last[:, :, 57:59]], dim=-1)

        end6 = time.time()
        print(f"Thời gian chạy sau net_out_val: {end6 - start6:.3f} giây")

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



    def test_demo(self, epoch, ap):

        print("inference")
        start1 = time.time()

        # Load names
        with open(self.args.code_path + '/scripts/EMAGE_2024/utils/assets/names_only.json', 'r') as f:
            names_data = json.load(f)
        names = names_data['name']  # should be a list of 51 items

        bvh_path = self.args.code_path + "/scripts/EMAGE_2024/utils/assets/1_fufu_0_1_1_first64.bvh"  
        bvh_pose_master = load_framewise_pose_file(bvh_path)     # shape: (F_bvh, j*3)

        print(f"abc")
        filename = 'result' # os.path.splitext(os.path.basename(ap))[0]
        
        # results_save_path = self.checkpoint_path + f"{epoch}/"
        results_save_path = "/Users/caocongdanh/Study/NCU/Lab/source/outputs/"
        if os.path.exists(results_save_path): 
            import shutil
            shutil.rmtree(results_save_path)
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        
        # test_demo = "/data/nas07/PersonalData/danh/EMAGEMixamo/mixamo_english_30_4_811_all/test_norm/bvh_full/"  
        test_demo = "inference"

        inference_model.eval()
        result_path = ""
        end1 = time.time()
        print(f"Thời gian chạy inference: {end1 - start1:.3f} giây")
        # print(f"len(self.test_loader.dataset): {len(self.test_loader.dataset)}")
        # print(f"len(self.test_loader): {len(self.test_loader)}")
        with torch.no_grad():
            # for its, batch_data in enumerate(self.test_loader):
            batch_data = next(iter(self.test_loader))

            start2 = time.time()
            loaded_data = self._load_data(batch_data)
            loaded_data['in_audio'] = ap[1]
            end2 = time.time()
            print(f"Thời gian chạy _load_data: {end2 - start2:.3f} giây")
            start = time.time()
            net_out = self._g_test_micro(loaded_data)
            end = time.time()
            print(f"Thời gian chạy _g_test_demo: {end - start:.3f} giây")
            print(f"in test_demo: {net_out.keys()}")
            print(f"_g_test output: {net_out}")
            start3 = time.time()
            tar_pose = net_out['tar_pose']
            rec_pose = net_out['rec_pose']
            tar_exps = net_out['tar_exps']
            rec_trans = net_out['rec_trans']
            tar_trans = net_out['tar_trans']
            rec_exps = net_out['rec_exps']

            # audio from micro
            in_audio = ap

            bs, n, j = tar_pose.shape[0], tar_pose.shape[1], _joints

            rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*64, j, 6))
            rec_pose = rc.matrix_to_euler_angles(rec_pose, "YXZ").reshape(bs, 64, j, 3) 

            rec_pose = np.rad2deg(rec_pose.cpu().numpy())

            rec_pose = fix_abnormal_joint_motion_soft(
                rec_pose,
                joint_idx=27,
                fps=30.0,
                abnormal_threshold_deg_per_s=600.0,
                blend_factor=0.2  # tweak to control how much smoothing
            )
        
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
            
            trans = trans
            rec_pose = np.concatenate([trans, rec_pose], axis=1)  # (frames, 3 + j*3)
            
            total_length += n
                
            # seq_name = test_seq_list[its].split('.')[0]  # safer filename
            # result_path = f"{results_save_path}res_{filename}.bvh"
            # with open(f"{results_save_path}result_raw_{filename}.bvh", 'w+') as f_real:
            #     for line_id in range(rec_pose.shape[0]): #,args.pre_frames, args.pose_length
            #         line_data = np.array2string(rec_pose[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
            #         f_real.write(line_data[1:-2]+'\n')

            # === Save input audio as .wav ===
            # audio_np = in_audio.squeeze().cpu().numpy()
            # audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-8)  # normalize to [-1, 1]
            # audio_int16 = (audio_np * 32767).astype(np.int16)
            # sample_rate = getattr(self.args, "audio_sample_rate", 16000)
            # wavfile.write(f"{results_save_path}in_audio_{filename}.wav", sample_rate, audio_int16)

            ## Save face
            rec_exps = rec_exps.cpu().numpy().reshape(bs*64, 51)
            rec_exps = np.clip(rec_exps, 0, 1) # Clip the values to range of 0-1
            frames = [{"weights": list(map(float, frame))} for frame in rec_exps]
        
            save_dict = {
                "name": names,
                "frames": frames
            }
        
            # save_path = f"{results_save_path}face_{filename}.json"
            # with open(save_path, 'w') as f:
            #     json.dump(save_dict, f, indent=2)
            
            end3 = time.time()
            print(f"Thời gian chạy sau _g_test_demo: {end3 - start3:.3f} giây")
            del rec_pose, tar_pose, rec_trans, net_out, loaded_data, in_audio
            torch.cuda.empty_cache()

            # data_tools.result2target_vis("mixamo_joint_full", results_save_path, results_save_path, test_demo, mode="all_or_lower", verbose=False)
        # result = gr.Video(value=render_vid_path, visible=True)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")

        return 'result_path'

import threading
import queue
import numpy as np
import sounddevice as sd
import torch
import warnings
import sys
import resampy
import librosa

@logger.catch
def emage_from_mic():

    print(sd.query_devices())
    print(int(sd.query_devices(sd.default.device[1])['default_samplerate']))
    # line898

    smplx_path = None
    text_path = None
    args = config.parse_args()

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    other_tools_hf.set_random_seed(args)
    other_tools_hf.print_exp_info(args)

    target_sr = 16000
    chunk_duration = 64 / 30
    chunk_samples = int(target_sr * chunk_duration)

    audio_queue = queue.Queue()
    buffer = np.empty((0, 1), dtype='float32')
    buffer_lock = threading.Lock()  # bảo vệ buffer khi nhiều thread truy cập

    # Callback micro → queue
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)
        mic_sr = int(sd.query_devices(sd.default.device[1])['default_samplerate'])
        if mic_sr != target_sr:
            indata_resampled = resampy.resample(indata[:, 0], mic_sr, target_sr)[:, np.newaxis]
        else:
            indata_resampled = indata
        # print(f"Callback nhận {frames} frames, shape={indata.shape}")
        # print(f"Callback nhận {frames} samples")
        audio_queue.put(indata_resampled.copy())
        # print("Callback pushed:", audio_queue.qsize())

    # Chuẩn bị model
    data, samplerate = sf.read("/Users/caocongdanh/Downloads/delete/1_fufu_0_1_1.wav")
    input_sample = (target_sr, data)
    trainer = BaseTrainer(args, sp=smplx_path, ap=input_sample, tp=text_path)
    if not _global_inference:
        other_tools_hf.load_checkpoints(inference_model, test_checkpoint, args.g_name)

    device = next(inference_model.parameters()).device
    print("Bắt đầu ghi âm từ micro, mỗi ~2s sẽ inference...")

    def read_audio_thread():
        nonlocal buffer
        while True:
            if not audio_queue.empty():
                chunk = audio_queue.get()
                with buffer_lock:
                    buffer = np.concatenate((buffer, chunk), axis=0)
                    # print(f"Buffer hiện có {buffer.shape[0]} samples")

    def inference_thread():
        nonlocal buffer
        while True:
            # print(f"Buffer hiện có {buffer.shape[0]} samples, cần {chunk_samples}")
            with buffer_lock:
                if buffer.shape[0] >= chunk_samples:
                    chunk = buffer[:chunk_samples]
                    buffer = buffer[chunk_samples:]
                else:
                    chunk = None
            if chunk is not None:
                in_audio_tensor = torch.tensor(chunk.squeeze(-1).T, dtype=torch.float32).unsqueeze(0).to(device)
                audio_input = (target_sr, in_audio_tensor)
                # preprocessing audio
                audio_file = audio_input[1]
                sr = audio_input[0]
                audio_each_file = audio_file.detach().cpu().numpy().astype(np.float32)
                audio_each_file = librosa.resample(audio_each_file, orig_sr=sr, target_sr=target_sr)
                audio_each_file = torch.from_numpy(audio_each_file).to(device)
                audio_input = (target_sr, audio_each_file)
                start = time.time()
                result = trainer.test_demo(999, ap=audio_input)
                end = time.time()
                print(f"Thời gian chạy: {end - start:.3f} giây")
                print("Inference xong 1 chunk (~2s audio)")

    try:
        with sd.InputStream(samplerate=int(sd.query_devices(sd.default.device[1])['default_samplerate']), channels=1, blocksize=4096, dtype='float32', callback=audio_callback):
            print("Ghi âm đang chạy, nhấn Ctrl+C để dừng...")
            t1 = threading.Thread(target=read_audio_thread, daemon=True)
            t2 = threading.Thread(target=inference_thread, daemon=True)
            t1.start()
            t2.start()
            t1.join()  # giữ main thread
    except KeyboardInterrupt:
        print("Dừng ghi âm, xử lý phần audio còn lại...")
        with buffer_lock:
            if buffer.shape[0] > 0:
                in_audio_tensor = torch.tensor(buffer.squeeze(-1).T, dtype=torch.float32).unsqueeze(0).to(device)
                audio_input = (target_sr, in_audio_tensor)
                result = trainer.test_demo(999, ap=audio_input)
                print("Inference xong phần audio còn lại")
        print("Ghi âm đã kết thúc.")


            
if __name__ == "__main__":

    os.environ["MASTER_ADDR"] = '127.0.0.1'
    os.environ["MASTER_PORT"] = '8675'

    # while True: 
        # audio_path = input("Enter audio path (.wav): ").strip()

        # if audio_path == '0':
        #     break

        # if not os.path.isfile(audio_path):
        #     print(f"Error'{audio_path}' no exist")
        #     sys.exit(1)

        # output_file = emage(audio_path)
        # print(f"\nFinished. Result path: {output_file}\n")
    emage_from_mic()
    # _global_inference = True

