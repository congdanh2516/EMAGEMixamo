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

class CustomTrainer(train.BaseTrainer):
    def __init__(self, args, loader_type="train", build_cache=True): # adding build_cache=True
        super().__init__(args)
        self.args = args
        self.joints = self.train_data.joints
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["skeleton_face"]
        self.tar_joint_list_upper = joints_list["skeleton_upper"]
        self.tar_joint_list_hands = joints_list["skeleton_hand"]
        self.tar_joint_list_lower = joints_list["skeleton_lower"]

        self.joints = 75
       
        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
            
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_upper:
            if joint_name == "Hips":
                self.joint_mask_upper[3:6] = 1
            else:
                self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
            
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
            
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        self.tracker = other_tools.EpochTracker(["fid", "l1div", "bc", "rec", "trans", "vel", "transv", 'dis', 'gen', 'acc', 'transa', 'exp', 'lvd', 'mse', "cls", "rec_face", "latent", "cls_full", "cls_self", "cls_word", "latent_word","latent_self"], [False,True,True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False,False,False])
        
        
        vq_model_module = __import__(f"models.motion_representation", fromlist=["something"])
        self.args.vae_layer = 2
        self.args.vae_length = 256
        self.args.vae_test_dim = 6
        self.vq_model_face = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        # print(self.vq_model_face)
        logger.info(f"THIS IS FACE")
        other_tools.load_checkpoints(self.vq_model_face, "./outputs/audio2pose/custom/0205_173746_cnn_vqvae_face_30/last_1.bin", args.e_name)
        
        self.args.vae_test_dim = 39
        self.vq_model_upper = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_upper, "./outputs/audio2pose/custom/1203_213726_cnn_vqvae_upper_30_skeleton/last_1.bin", args.e_name)
        
        self.args.vae_test_dim = 144
        self.vq_model_hands = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        checkpoint = other_tools.load_checkpoints(self.vq_model_hands, "./outputs/audio2pose/custom/0224_214718_cnn_vqvae_hand_30/last_1.bin", args.e_name)
        
        logger.info(f"VQVAE Model Keys: {list(self.vq_model_hands.state_dict().keys())}")

        if checkpoint:
            logger.info(f"Checkpoint Keys: {list(checkpoint.keys())}")
        else:
            logger.warning("Checkpoint loading failed or returned None.")

        logger.info(f"TO HAND, BEGIN LOADING LOWER")
        
        
        self.args.vae_test_dim = 33
        self.args.vae_layer = 4
        self.vq_model_lower = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)

        logger.info(f"VQVAE: {self.vq_model_lower.keys}")
        logger.info(f"CHECK POINT : {other_tools.load_checkpoints(self.vq_model_hands, './outputs/audio2pose/custom/0224_214718_cnn_vqvae_lower_30/last_1.bin', args.e_name).keys()}")
        
        other_tools.load_checkpoints(self.vq_model_lower, "./outputs/audio2pose/custom/0206_192547_cnn_vqvae_lower_30/last_1.bin", args.e_name)

        logger.info(f"DONE LOAD LOWER")

        # remaining
        self.args.vae_test_dim = 61
        self.args.vae_layer = 4
        self.global_motion = getattr(vq_model_module, "VAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.global_motion, ".outputs/audio2pose/custom/0208_120714_cnn_vqvae_lower_foot_30/last_1.bin", args.e_name)
        
        self.args.vae_test_dim = 225
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
        
        #---------add new from BEAT----------#
        self.audio_fps = args.audio_fps
        self.loader_type = loader_type
        self.new_cache = args.new_cache
        self.pose_rep = args.pose_rep

        # build cache
        self.disable_filtering = args.disable_filtering
        self.clean_first_seconds = args.clean_first_seconds
        self.clean_final_seconds = args.clean_final_seconds

        # self.data_dir
        if loader_type == "train":
            self.data_dir = args.root_path + args.train_data_path
            self.multi_length_training = args.multi_length_training
        elif loader_type == "val":
            self.data_dir = args.root_path + args.val_data_path
            self.multi_length_training = args.multi_length_training 
        else:
            self.data_dir = args.root_path + args.test_data_path
            self.multi_length_training = [1.0]
      
        self.max_length = int(self.pose_length * self.multi_length_training[-1])
        
        if self.word_rep is not None:
            with open(f"{args.root_path}{args.train_data_path[:-6]}vocab.pkl", 'rb') as f:
                self.lang_model = pickle.load(f)
        
        preloaded_dir = self.data_dir + f"{self.pose_rep}_cache"

        #-------# need to confirm its usage
        self.mean_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/bvh_mean.npy")
        self.std_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/bvh_std.npy")
        self.audio_norm = args.audio_norm
        self.facial_norm = args.facial_norm
        if self.audio_norm:
            self.mean_audio = np.load(args.root_path+args.mean_pose_path+f"{args.audio_rep}/npy_mean.npy")
            self.std_audio = np.load(args.root_path+args.mean_pose_path+f"{args.audio_rep}/npy_std.npy")
        if self.facial_norm:
            self.mean_facial = np.load(args.root_path+args.mean_pose_path+f"{args.facial_rep}/json_mean.npy")
            self.std_facial = np.load(args.root_path+args.mean_pose_path+f"{args.facial_rep}/json_std.npy")
        #------#
        
        if build_cache and self.rank == 0:
            self.build_cache(preloaded_dir)
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"]


        #----cache_generation------#
        self.disable_filtering = args.disable_filtering
        
        #end-------add new from BEAT---------#  
    
    #---------add new from BEAT----------#

    #-begin-build_cache------------------#

    def build_cache(self, preloaded_dir):
        logger.info(f"Audio bit rate: {self.audio_fps}")
        logger.info("Reading data '{}'...".format(self.data_dir))
        
        # pose_length_extended = int(round(self.pose_length))
        logger.info("Creating the dataset cache...")
        if self.new_cache:
            if os.path.exists(preloaded_dir):
                shutil.rmtree(preloaded_dir)

        if os.path.exists(preloaded_dir):
            logger.info("Found the cache {}".format(preloaded_dir))
        elif self.loader_type == "test":
            self.cache_generation(
                preloaded_dir, True, 
                0, 0,
                is_test=True)
        else: 
            self.cache_generation(
                preloaded_dir, self.disable_filtering, 
                self.clean_first_seconds, self.clean_final_seconds,
                is_test=False)
            
    #-end-build_cache-------------------#   

    #-begin-cache_generation-------------------#

    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds,  clean_final_seconds, is_test=False):
        self.n_out_samples = 0
        pose_files = sorted(glob.glob(os.path.join(self.data_dir, f"{self.pose_rep}") + "/*.bvh"), key=str,)  
        # create db for samples

        ##******************Note: Change desired FPS at self.pose_fps/xx*******************
        map_size = int(1024 * 1024 * 2048 * (self.audio_fps/16000)**3 * 4) * (len(pose_files)/30*(self.pose_fps/30)) * len(self.multi_length_training) * self.multi_length_training[-1] * 2 # in 1024 MB
        dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        n_filtered_out = defaultdict(int)
    
        for pose_file in pose_files:
            pose_each_file = []
            pose_each_file_new = []
            audio_each_file = []
            facial_each_file = []
            word_each_file = []
            emo_each_file = []
            sem_each_file = []
            vid_each_file = []
            
            id_pose = pose_file.split("/")[-1][:-4] #1_wayne_0_1_1
            logger.info(colored(f"# ---- Building cache for Pose   {id_pose} ---- #", "blue"))
            
            with open(pose_file, "r") as pose_data:
                for j, line in enumerate(pose_data.readlines()):
                    data = np.fromstring(line, dtype=float, sep=" ") # 1*27 e.g., 27 rotation 
                    data =  np.array(data)
                    # logger.info(f"DATA: {data} {len(data)}")
                    # logger.info(f"JOINT MASK: {self.joint_mask} {len(self.joint_mask)}")
                    data = data * self.joint_mask
                    data = data[self.joint_mask.astype(bool)]
                    pose_each_file.append(data)

            #     print("X1: ", len(pose_each_file_new))
                
            # print("joint_mask: ", len(self.joint_mask))
            # print("joint_mask: ", self.joint_mask)

            # assert 120%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 120'
            # stride = int(120/self.args.pose_fps)
            # with open(pose_file, "r") as pose_data:
            #     for j, line in enumerate(pose_data.readlines()):
            #         # if j < 431: continue     
            #         # if j%stride != 0:continue
            #         data = np.fromstring(line, dtype=float, sep=" ")

            #         rot_data = rc.euler_angles_to_matrix(torch.from_numpy(np.deg2rad(data)).reshape(-1, self.joints,3), "XYZ")
            #         rot_data = rc.matrix_to_axis_angle(rot_data).reshape(-1, self.joints*3) 
            #         rot_data = rot_data.numpy() * self.joint_mask
            #         rot_data = rot_data[:, self.joint_mask.astype(bool)]
            #         pose_each_file.append(rot_data)

            pose_each_file = np.array(pose_each_file) # n frames * 27
            # shape_each_file = np.repeat(np.array(-1).reshape(1, 1), pose_each_file.shape[0], axis=0)

            if self.audio_rep is not None:
                logger.info(f"# ---- Building cache for Audio  {id_pose} and Pose {id_pose} ---- #")
                audio_file = pose_file.replace(self.pose_rep, self.audio_rep).replace("bvh", "npy")
                try:
                    audio_each_file = np.load(audio_file)
                except:
                    logger.warning(f"# ---- file not found for Audio {id_pose}, skip all files with the same id ---- #")
                    continue
                if self.audio_norm: 
                    audio_each_file = (audio_each_file - self.mean_audio) / self.std_audio
                    
            if self.facial_rep is not None:
                logger.info(f"# ---- Building cache for Facial {id_pose} and Pose {id_pose} ---- #")
                facial_file = pose_file.replace(self.pose_rep, self.facial_rep).replace("bvh", "json")
                try:
                    with open(facial_file, 'r') as facial_data_file:
                        facial_data = json.load(facial_data_file)
                        for j, frame_data in enumerate(facial_data['frames']):
                            if self.facial_norm:
                                facial_each_file.append((frame_data['weights']-self.mean_facial) / self.std_facial)
                            else:
                                facial_each_file.append(frame_data['weights'])
                    facial_each_file = np.array(facial_each_file)
                except:
                    logger.warning(f"# ---- file not found for Facial {id_pose}, skip all files with the same id ---- #")
                    continue
                    
            if id_pose.split("_")[-1] == "b":
                time_offset = 30 if int(id_pose.split("_")[-3]) % 2 == 0 else 300
                logger.warning(time_offset)
            else:
                time_offset = 0
                
            if self.word_rep is not None:
                logger.info(f"# ---- Building cache for Word   {id_pose} and Pose {id_pose} ---- #")
                word_file = pose_file.replace(self.pose_rep, self.word_rep).replace("bvh", "TextGrid")
                try:
                    tgrid = tg.TextGrid.fromFile(word_file)
                except:
                    logger.warning(f"# ---- file not found for Word {id_pose}, skip all files with the same id ---- #")
                    continue
                # the length of text file are reduce to the length of motion file, for x_x_a or x_x_b
                for i in range(pose_each_file.shape[0]):
                    found_flag = False
                    current_time = i/self.pose_fps + time_offset
                    for word in tgrid[0]:
                        word_n, word_s, word_e = word.mark, word.minTime, word.maxTime
                        if word_s<=current_time and current_time<=word_e:
                            if word_n == " ":
                                #TODO now don't have eos and sos token
                                word_each_file.append(self.lang_model.PAD_token)
                            else:    
                                word_each_file.append(self.lang_model.get_word_index(word_n))
                            found_flag = True
                            break
                        else: continue   
                    if not found_flag: word_each_file.append(self.lang_model.UNK_token)
                # list of index
                word_each_file = np.array(word_each_file)
                    
            if self.emo_rep is not None:
                logger.info(f"# ---- Building cache for Emo    {id_pose} and Pose {id_pose} ---- #")
                emo_file = pose_file.replace(self.pose_rep, self.emo_rep).replace("bvh", "csv")
                try:    
                    emo_all = pd.read_csv(emo_file, 
                        sep=',', 
                        names=["name", "start_time", "end_time", "duration", "score"])
                except:
                    logger.warning(f"# ---- file not found for Emo {id_pose}, skip all files with the same id ---- #")
                    continue
                for i in range(pose_each_file.shape[0]):
                    found_flag = False
                    for j, (start, end, score) in enumerate(zip(emo_all['start_time'],emo_all['end_time'], emo_all['score'])):
                        current_time = i/self.pose_fps + time_offset
                        if start<=current_time and current_time<=end: 
                            emo_each_file.append(score)
                            found_flag=True
                            break
                        else: continue 
                    if not found_flag: emo_each_file.append(0)
                emo_each_file = np.array(emo_each_file)
                #print(emo_each_file)
                
            if self.sem_rep is not None:
                logger.info(f"# ---- Building cache for Sem    {id_pose} and Pose {id_pose} ---- #")
                sem_file = pose_file.replace(self.pose_rep, self.sem_rep).replace("bvh", "txt")
                try:
                    sem_all = pd.read_csv(sem_file, 
                        sep='\t', 
                        names=["name", "start_time", "end_time", "duration", "score", "keywords"])
                except:
                    logger.warning(f"# ---- file not found for Sem {id_pose}, skip all files with the same id ---- #")
                    continue
                # we adopt motion-level semantic score here. 
                for i in range(pose_each_file.shape[0]):
                    found_flag = False
                    for j, (start, end, score) in enumerate(zip(sem_all['start_time'],sem_all['end_time'], sem_all['score'])):
                        current_time = i/self.pose_fps + time_offset
                        if start<=current_time and current_time<=end: 
                            sem_each_file.append(score)
                            found_flag=True
                            break
                        else: continue 
                    if not found_flag: sem_each_file.append(0.)
                sem_each_file = np.array(sem_each_file)
                #print(sem_each_file)
            
            ##------------Modify here-------## 
            if self.id_rep is not None:
                vid_each_file.append(int(id_pose.split("_")[0])-1)

            ## Note: it's only need to mapping if there is missing file in the dataset
                ## REMOVE ABOVE LINE BY THESE LINE IF NEED MAPPING
                # int_value = self.idmapping(int(f_name.split("_")[0]))
                # vid_each_file = np.repeat(np.array(int_value).reshape(1, 1), pose_each_file.shape[0], axis=0)
            
            ##------------------------------##
            
            filtered_result = self._sample_from_clip(
                dst_lmdb_env,
                audio_each_file, pose_each_file, facial_each_file, word_each_file,
                vid_each_file, emo_each_file, sem_each_file,
                disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
                ) 
            for type in filtered_result.keys():
                n_filtered_out[type] += filtered_result[type]
                                
        with dst_lmdb_env.begin() as txn:
            logger.info(colored(f"no. of samples: {txn.stat()['entries']}", "cyan"))
            n_total_filtered = 0
            for type, n_filtered in n_filtered_out.items():
                logger.info("{}: {}".format(type, n_filtered))
                n_total_filtered += n_filtered
            # logger.info(colored("no. of excluded samples: {} ({:.1f}%)".format(
            #     n_total_filtered, 100 * n_total_filtered / (txn.stat()["entries"] + n_total_filtered)), "cyan"))
        dst_lmdb_env.sync()
        dst_lmdb_env.close()
        
    #-end-cache_generation-------------------#

    #-begin-normalize_pose---------------#

    @staticmethod
    def normalize_pose(dir_vec, mean_pose, std_pose=None, joint_mask=None):
        mean_pose = mean_pose[joint_mask.astype(bool)]
        std_pose = std_pose[joint_mask.astype(bool)]

        return (dir_vec - mean_pose) / std_pose 

    #-end-normalize_pose---------------#
        
    #end-------add new from BEAT---------#
    
    
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    # def _load_data(self, dict_data):
    #     tar_pose_raw = dict_data["pose"]
    #     tar_pose = tar_pose_raw[:, :, :165].to(self.rank)
    #     tar_contact = tar_pose_raw[:, :, 165:169].to(self.rank)
    #     tar_trans = dict_data["trans"].to(self.rank)
    #     tar_exps = dict_data["facial"].to(self.rank)
    #     in_audio = dict_data["audio"].to(self.rank) 
    #     in_word = dict_data["word"].to(self.rank)
    #     tar_beta = dict_data["beta"].to(self.rank)
    #     tar_id = dict_data["id"].to(self.rank).long()
    #     bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

    #     tar_pose_jaw = tar_pose[:, :, 66:69]
    #     tar_pose_jaw = rc.axis_angle_to_matrix(tar_pose_jaw.reshape(bs, n, 1, 3))
    #     tar_pose_jaw = rc.matrix_to_rotation_6d(tar_pose_jaw).reshape(bs, n, 1*6)
    #     tar_pose_face = torch.cat([tar_pose_jaw, tar_exps], dim=2)

    #     tar_pose_hands = tar_pose[:, :, 25*3:55*3]
    #     tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
    #     tar_pose_hands = rc.matri
    #     x_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

    #     tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
    #     tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
    #     tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

    #     tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
    #     tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
    #     tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
    #     tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)

    #     # tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
    #     # tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
    #     tar4dis = torch.cat([tar_pose_jaw, tar_pose_upper, tar_pose_hands, tar_pose_leg], dim=2)

    #     tar_index_value_face_top = self.vq_model_face.map2index(tar_pose_face) # bs*n/4
    #     tar_index_value_upper_top = self.vq_model_upper.map2index(tar_pose_upper) # bs*n/4
    #     tar_index_value_hands_top = self.vq_model_hands.map2index(tar_pose_hands) # bs*n/4
    #     tar_index_value_lower_top = self.vq_model_lower.map2index(tar_pose_lower) # bs*n/4
      
    #     latent_face_top = self.vq_model_face.map2latent(tar_pose_face) # bs*n/4
    #     latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper) # bs*n/4
    #     latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands) # bs*n/4
    #     latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower) # bs*n/4
        
    #     latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2)
        
    #     index_in = torch.stack([tar_index_value_upper_top, tar_index_value_hands_top, tar_index_value_lower_top], dim=-1).long()
        
    #     tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
    #     tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 55*6)
    #     latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
    #     # print(tar_index_value_upper_top.shape, index_in.shape)
    #     return {
    #         "tar_pose_jaw": tar_pose_jaw,
    #         "tar_pose_face": tar_pose_face,
    #         "tar_pose_upper": tar_pose_upper,
    #         "tar_pose_lower": tar_pose_lower,
    #         "tar_pose_hands": tar_pose_hands,
    #         'tar_pose_leg': tar_pose_leg,
    #         "in_audio": in_audio,
    #         "in_word": in_word,
    #         "tar_trans": tar_trans,
    #         "tar_exps": tar_exps,
    #         "tar_beta": tar_beta,
    #         "tar_pose": tar_pose,
    #         "tar4dis": tar4dis,
    #         "tar_index_value_face_top": tar_index_value_face_top,
    #         "tar_index_value_upper_top": tar_index_value_upper_top,
    #         "tar_index_value_hands_top": tar_index_value_hands_top,
    #         "tar_index_value_lower_top": tar_index_value_lower_top,
    #         "latent_face_top": latent_face_top,
    #         "latent_upper_top": latent_upper_top,
    #         "latent_hands_top": latent_hands_top,
    #         "latent_lower_top": latent_lower_top,
    #         "latent_in":  latent_in,
    #         "index_in": index_in,
    #         "tar_id": tar_id,
    #         "latent_all": latent_all,
    #         "tar_pose_6d": tar_pose_6d,
    #         "tar_contact": tar_contact,
    #     }

    def _load_data(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pyarrow.deserialize(sample)
            tar_pose, in_audio, in_facial, in_word, emo, sem, vid = sample
            vid = torch.from_numpy(vid).int()
            emo = torch.from_numpy(emo).int()
            sem = torch.from_numpy(sem).float() 
            in_audio = torch.from_numpy(in_audio).float() 
            in_word = torch.from_numpy(in_word).int()  
            if self.loader_type == "test":
                tar_pose = torch.from_numpy(tar_pose).float()
                in_facial = torch.from_numpy(in_facial).float()
                            
            else:
                tar_pose = torch.from_numpy(tar_pose).reshape((tar_pose.shape[0], -1)).float()
                in_facial = torch.from_numpy(in_facial).reshape((in_facial.shape[0], -1)).float()
            return {"pose":tar_pose, "audio":in_audio, "facial":in_facial, "word":in_word, "id":vid, "emo":emo, "sem":sem}

    
    
    def _g_training(self, loaded_data, use_adv, mode="train", epoch=0):
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        # ------ full generatation task ------ #
        mask_val = torch.ones(bs, n, self.args.pose_dims+3+4).float().cuda()
        mask_val[:, :self.args.pre_frames, :] = 0.0
        
        net_out_val  = self.model(
            loaded_data['in_audio'], loaded_data['in_word'], mask=mask_val,
            in_id = loaded_data['tar_id'], in_motion = loaded_data['latent_all'],
            use_attentions = True)
        g_loss_final = 0
        loss_latent_face = self.reclatent_loss(net_out_val["rec_face"], loaded_data["latent_face_top"]) # reclatent_loss is MSELoss
        loss_latent_lower = self.reclatent_loss(net_out_val["rec_lower"], loaded_data["latent_lower_top"])
        loss_latent_hands = self.reclatent_loss(net_out_val["rec_hands"], loaded_data["latent_hands_top"])
        loss_latent_upper = self.reclatent_loss(net_out_val["rec_upper"], loaded_data["latent_upper_top"])
        loss_latent = self.args.lf*loss_latent_face + self.args.ll*loss_latent_lower + self.args.lh*loss_latent_hands + self.args.lu*loss_latent_upper
        self.tracker.update_meter("latent", "train", loss_latent.item())
        g_loss_final += loss_latent

        rec_index_face_val = self.log_softmax(net_out_val["cls_face"]).reshape(-1, self.args.vae_codebook_size)
        rec_index_upper_val = self.log_softmax(net_out_val["cls_upper"]).reshape(-1, self.args.vae_codebook_size)
        rec_index_lower_val = self.log_softmax(net_out_val["cls_lower"]).reshape(-1, self.args.vae_codebook_size)
        rec_index_hands_val = self.log_softmax(net_out_val["cls_hands"]).reshape(-1, self.args.vae_codebook_size)
        tar_index_value_face_top = loaded_data["tar_index_value_face_top"].reshape(-1)
        tar_index_value_upper_top = loaded_data["tar_index_value_upper_top"].reshape(-1)
        tar_index_value_lower_top = loaded_data["tar_index_value_lower_top"].reshape(-1)
        tar_index_value_hands_top = loaded_data["tar_index_value_hands_top"].reshape(-1)    
        loss_cls = self.args.cf*self.cls_loss(rec_index_face_val, tar_index_value_face_top)\
            + self.args.cu*self.cls_loss(rec_index_upper_val, tar_index_value_upper_top)\
            + self.args.cl*self.cls_loss(rec_index_lower_val, tar_index_value_lower_top)\
            + self.args.ch*self.cls_loss(rec_index_hands_val, tar_index_value_hands_top)
        self.tracker.update_meter("cls_full", "train", loss_cls.item())
        g_loss_final += loss_cls 
        
        if mode == 'train':
        #     # ------ masked gesture moderling------ #
            mask_ratio = (epoch / self.args.epochs) * 0.95 + 0.05  
            mask = torch.rand(bs, n, self.args.pose_dims+3+4) < mask_ratio
            mask = mask.float().cuda()
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
            
            # ------ masked audio gesture moderling ------ #
            net_out_word  = self.model(
                loaded_data['in_audio'], loaded_data['in_word'], mask=mask,
                in_id = loaded_data['tar_id'], in_motion = loaded_data['latent_all'],
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

        if mode != 'train':
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

            rec_pose_jaw = rec_face[:, :, :6]
            rec_pose_legs = rec_lower[:, :, :54]

            rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
            rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
            rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 13*3)
            rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
            
            rec_pose_lower = rec_pose_legs.reshape(bs, n, 12, 6)
            rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
            rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
            rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
            
            rec_pose_hands = rec_hands.reshape(bs, n, 48, 6)
            rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
            rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
            rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
            
            rec_pose_jaw = rec_pose_jaw.reshape(bs*n, 6)
            rec_pose_jaw = rc.rotation_6d_to_matrix(rec_pose_jaw)
            rec_pose_jaw = rc.matrix_to_axis_angle(rec_pose_jaw).reshape(bs*n, 1*3)
            rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
            rec_pose[:, 66:69] = rec_pose_jaw
            # print(rec_pose.shape, tar_pose.shape)
            
            # tar_trans = loaded_data["tar_trans"]
            # rec_trans_v_s = rec_lower[:, :, 54:57]
            # rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
            # rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
            # rec_y_trans = rec_trans_v_s[:,:,1:2]
            # rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
            # tar_pose = loaded_data["tar_pose"]
            # tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
            # tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
            rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, j, 3))
            rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)


        if mode == 'train':
            return g_loss_final
        elif mode == 'val':
            return {
                'rec_pose': rec_pose,
                # rec_trans': rec_pose_trans,
                'tar_pose': loaded_data["tar_pose_6d"],
            }
        else:
            return {
                'rec_pose': rec_pose,
                # 'rec_trans': rec_trans,
                'tar_pose': loaded_data["tar_pose"],
                'tar_exps': loaded_data["tar_exps"],
                'tar_beta': loaded_data["tar_beta"],
                'tar_trans': loaded_data["tar_trans"],
                # 'rec_exps': rec_exps,
            }
    

    def _g_test(self, loaded_data):
        mode = 'test'
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        tar_pose = loaded_data["tar_pose"]
        tar_beta = loaded_data["tar_beta"]
        in_word = loaded_data["in_word"]
        tar_exps = loaded_data["tar_exps"]
        tar_contact = loaded_data["tar_contact"]
        in_audio = loaded_data["in_audio"]
        tar_trans = loaded_data["tar_trans"]

        remain = n%8
        if remain != 0:
            tar_pose = tar_pose[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            in_word = in_word[:, :-remain]
            tar_exps = tar_exps[:, :-remain, :]
            tar_contact = tar_contact[:, :-remain, :]
            n = n - remain

        tar_pose_jaw = tar_pose[:, :, 66:69]
        tar_pose_jaw = rc.axis_angle_to_matrix(tar_pose_jaw.reshape(bs, n, 1, 3))
        tar_pose_jaw = rc.matrix_to_rotation_6d(tar_pose_jaw).reshape(bs, n, 1*6)
        tar_pose_face = torch.cat([tar_pose_jaw, tar_exps], dim=2)

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)
        
        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 55*6)
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
        
        # pad latent_all_9 to the same length with latent_all
        # if n - latent_all_9.shape[1] >= 0:
        #     latent_all = torch.cat([latent_all_9, torch.zeros(bs, n - latent_all_9.shape[1], latent_all_9.shape[2]).cuda()], dim=1)
        # else:
        #     latent_all = latent_all_9[:, :n, :]

        for i in range(0, roundt):
            in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            # audio fps is 16000 and pose fps is 30
            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames]
            in_id_tmp = loaded_data['tar_id'][:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+4).float().cuda()
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
                in_motion = latent_all_tmp,
                in_id = in_id_tmp,
                use_attentions=True,)
            
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
            # if self.args.cf != 0:
            #     rec_face_last = self.vq_model_face.decode(rec_index_face)
            # else:
            #     rec_face_last = self.vq_model_face.decoder(rec_index_face)

            rec_pose_legs = rec_lower_last[:, :, :54]
            bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
            rec_pose_upper = rec_upper_last.reshape(bs, n, 13, 6)
            rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
            rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 13*3)
            rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
            rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
            rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
            rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
            rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
            rec_pose_hands = rec_hands_last.reshape(bs, n, 30, 6)
            rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
            rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
            rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
            rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
            rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, j, 3))
            rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
            rec_trans_v_s = rec_lower_last[:, :, 54:57]
            rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
            rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
            rec_y_trans = rec_trans_v_s[:,:,1:2]
            rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
            latent_last = torch.cat([rec_pose, rec_trans, rec_lower_last[:, :, 57:61]], dim=-1)

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

        rec_exps = rec_face[:, :, 6:]
        rec_pose_jaw = rec_face[:, :, :6]
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_jaw.shape[0], rec_pose_jaw.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 13*3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = rc.matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9*6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
        rec_pose_jaw = rec_pose_jaw.reshape(bs*n, 6)
        rec_pose_jaw = rc.rotation_6d_to_matrix(rec_pose_jaw)
        rec_pose_jaw = rc.matrix_to_axis_angle(rec_pose_jaw).reshape(bs*n, 1*3)
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
        rec_pose[:, 66:69] = rec_pose_jaw

        to_global = rec_lower
        to_global[:, :, 54:57] = 0.0
        to_global[:, :, :54] = rec_lower2global
        rec_global = self.global_motion(to_global)

        rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
        rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
        rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
        rec_y_trans = rec_trans_v_s[:,:,1:2]
        rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]

        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs*n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
        tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs*n, j, 3))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
        
        return {
            'rec_pose': rec_pose,
            'rec_trans': rec_trans,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            'tar_beta': tar_beta,
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
        self.model.eval()
        # self.d_model.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.train_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_training(loaded_data, False, 'val', epoch)
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
                latent_out = self.eval_copy.map2latent(rec_pose).reshape(-1, self.args.vae_length).cpu().numpy()
                latent_ori = self.eval_copy.map2latent(tar_pose).reshape(-1, self.args.vae_length).cpu().numpy()
                if its == 0:
                    latent_out_motion_all = latent_out
                    latent_ori_all = latent_ori
                else:
                    latent_out_motion_all = np.concatenate([latent_out_motion_all, latent_out], axis=0)                 
                    latent_ori_all = np.concatenate([latent_ori_all, latent_ori], axis=0)
                if self.args.debug:
                    if its == 1: break
        fid_motion = data_tools.FIDCalculator.frechet_distance(latent_out_motion_all, latent_ori_all)
        self.tracker.update_meter("fid", "val", fid_motion)
        self.val_recording(epoch) 
    
    
    def test(self, epoch):
        
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        align = 0 
        latent_out = []
        latent_ori = []
        l2_all = 0 
        lvel = 0
        self.model.eval()
        self.smplx.eval()
        self.eval_copy.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                # print(rec_pose.shape, tar_pose.shape)
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                if (30/self.args.pose_fps) != 1:
                    assert 30%self.args.pose_fps == 0
                    n *= int(30/self.args.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                
                # print(rec_pose.shape, tar_pose.shape)
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
                remain = n%self.args.vae_test_len
                latent_out.append(self.eval_copy.map2latent(rec_pose[:, :n-remain]).reshape(-1, self.args.vae_length).detach().cpu().numpy()) # bs * n/8 * 240
                latent_ori.append(self.eval_copy.map2latent(tar_pose[:, :n-remain]).reshape(-1, self.args.vae_length).detach().cpu().numpy())
                
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                
                vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100)-tar_exps.reshape(bs*n, 100),
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3], 
                        left_hand_pose=rec_pose[:,25*3:40*3], 
                        right_hand_pose=rec_pose[:,40*3:55*3], 
                        return_joints=True, 
                        leye_pose=rec_pose[:, 69:72], 
                        reye_pose=rec_pose[:, 72:75],
                    )
                # vertices_tar = self.smplx(
                #         betas=tar_beta.reshape(bs*n, 300), 
                #         transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                #         expression=tar_exps.reshape(bs*n, 100)-tar_exps.reshape(bs*n, 100),
                #         jaw_pose=tar_pose[:, 66:69], 
                #         global_orient=tar_pose[:,:3], 
                #         body_pose=tar_pose[:,3:21*3+3], 
                #         left_hand_pose=tar_pose[:,25*3:40*3], 
                #         right_hand_pose=tar_pose[:,40*3:55*3], 
                #         return_joints=True, 
                #         leye_pose=tar_pose[:, 69:72], 
                #         reye_pose=tar_pose[:, 72:75],
                #     )
                vertices_rec_face = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                        expression=rec_exps.reshape(bs*n, 100), 
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3]-rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3]-rec_pose[:,3:21*3+3],
                        left_hand_pose=rec_pose[:,25*3:40*3]-rec_pose[:,25*3:40*3],
                        right_hand_pose=rec_pose[:,40*3:55*3]-rec_pose[:,40*3:55*3],
                        return_verts=True, 
                        return_joints=True,
                        leye_pose=rec_pose[:, 69:72]-rec_pose[:, 69:72],
                        reye_pose=rec_pose[:, 72:75]-rec_pose[:, 72:75],
                    )
                vertices_tar_face = self.smplx(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100), 
                    jaw_pose=tar_pose[:, 66:69], 
                    global_orient=tar_pose[:,:3]-tar_pose[:,:3],
                    body_pose=tar_pose[:,3:21*3+3]-tar_pose[:,3:21*3+3], 
                    left_hand_pose=tar_pose[:,25*3:40*3]-tar_pose[:,25*3:40*3],
                    right_hand_pose=tar_pose[:,40*3:55*3]-tar_pose[:,40*3:55*3],
                    return_verts=True, 
                    return_joints=True,
                    leye_pose=tar_pose[:, 69:72]-tar_pose[:, 69:72],
                    reye_pose=tar_pose[:, 72:75]-tar_pose[:, 72:75],
                )  
                joints_rec = vertices_rec["joints"].detach().cpu().numpy().reshape(1, n, 127*3)[0, :n, :55*3]
                # joints_tar = vertices_tar["joints"].detach().cpu().numpy().reshape(1, n, 127*3)[0, :n, :55*3]
                facial_rec = vertices_rec_face['vertices'].reshape(1, n, -1)[0, :n]
                facial_tar = vertices_tar_face['vertices'].reshape(1, n, -1)[0, :n]
                face_vel_loss = self.vel_loss(facial_rec[1:, :] - facial_tar[:-1, :], facial_tar[1:, :] - facial_tar[:-1, :])
                l2 = self.reclatent_loss(facial_rec, facial_tar)
                l2_all += l2.item() * n
                lvel += face_vel_loss.item() * n
                
                _ = self.l1_calculator.run(joints_rec)
                if self.alignmenter is not None:
                    in_audio_eval, sr = librosa.load(self.args.data_path+"wave16k/"+test_seq_list.iloc[its]['id']+".wav")
                    in_audio_eval = librosa.resample(in_audio_eval, orig_sr=sr, target_sr=self.args.audio_sr)
                    a_offset = int(self.align_mask * (self.args.audio_sr / self.args.pose_fps))
                    onset_bt = self.alignmenter.load_audio(in_audio_eval[:int(self.args.audio_sr / self.args.pose_fps*n)], a_offset, len(in_audio_eval)-a_offset, True)
                    beat_vel = self.alignmenter.load_pose(joints_rec, self.align_mask, n-self.align_mask, 30, True)
                    # print(beat_vel)
                    align += (self.alignmenter.calculate_align(onset_bt, beat_vel, 30) * (n-2*self.align_mask))
               
                tar_pose_np = tar_pose.detach().cpu().numpy()
                rec_pose_np = rec_pose.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
                rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100) 
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(bs*n, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(bs*n, 3)
                gt_npz = np.load(self.args.data_path+self.args.pose_rep +"/"+test_seq_list.iloc[its]['id']+".npz", allow_pickle=True)
                np.savez(results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=tar_pose_np,
                    expressions=tar_exp_np,
                    trans=tar_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 ,
                )
                np.savez(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                total_length += n

        logger.info(f"l2 loss: {l2_all/total_length}")
        logger.info(f"lvel loss: {lvel/total_length}")

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)
        fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fid score: {fid}")
        self.test_recording("fid", fid, epoch) 
        
        align_avg = align/(total_length-2*len(self.test_loader)*self.align_mask)
        logger.info(f"align score: {align_avg}")
        self.test_recording("bc", align_avg, epoch)

        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        self.test_recording("l1div", l1div, epoch)

        # data_tools.result2target_vis(self.args.pose_version, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")


    def test_demo(self, epoch):
        '''
        input audio and text, output motion
        do not calculate loss and metric
        save video
        '''
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        align = 0 
        latent_out = []
        latent_ori = []
        l2_all = 0 
        lvel = 0
        self.model.eval()
        self.smplx.eval()
        # self.eval_copy.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                # print(rec_pose.shape, tar_pose.shape)
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

                # interpolate to 30fps  
                if (30/self.args.pose_fps) != 1:
                    assert 30%self.args.pose_fps == 0
                    n *= int(30/self.args.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                
                # print(rec_pose.shape, tar_pose.shape)
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)

                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                        
                tar_pose_np = tar_pose.detach().cpu().numpy()
                rec_pose_np = rec_pose.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
                rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100) 
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(bs*n, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(bs*n, 3)

                gt_npz = np.load(self.args.data_path+self.args.pose_rep +"/"+test_seq_list.iloc[its]['id']+".npz", allow_pickle=True)
                np.savez(results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=tar_pose_np,
                    expressions=tar_exp_np,
                    trans=tar_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 ,
                )
                np.savez(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                total_length += n

        data_tools.result2target_vis(self.args.pose_version, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")
