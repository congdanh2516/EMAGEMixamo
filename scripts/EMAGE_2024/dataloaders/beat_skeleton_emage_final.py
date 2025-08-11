import os
import pickle
import math
import shutil
import numpy as np
import lmdb as lmdb
import textgrid as tg
import pandas as pd
import torch
import glob
import json
from termcolor import colored
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist
import pyarrow
from sklearn.preprocessing import normalize
import librosa 
import scipy.io.wavfile
from scipy import signal
from .build_vocab import Vocab
from .utils import rotation_conversions as rc
from .data_tools import joints_list
from .utils import other_tools
import stat

    
class CustomDataset(Dataset):
    def __init__(self, args, loader_type, augmentation=None, kwargs=None, build_cache=True):
        self.loader_type = loader_type
        self.rank = dist.get_rank()
        self.new_cache = args.new_cache
        self.pose_length = args.pose_length #34
        self.stride = args.stride #10
        self.pose_fps = args.pose_fps #15
        self.pose_dims = args.pose_dims # 141
        self.facial_dims = args.facial_dims

        self.args = args

        self.speaker_dims = args.speaker_dims
        self.loader_type = loader_type
        self.audio_rep = args.audio_rep
        self.pose_rep = args.pose_rep
        self.facial_rep = args.facial_rep
        self.word_rep = args.word_rep
        self.emo_rep = args.emo_rep
        self.sem_rep = args.sem_rep
        self.audio_fps = args.audio_fps
        self.id_rep = args.speaker_id
        
        self.disable_filtering = args.disable_filtering
        self.clean_first_seconds = args.clean_first_seconds
        self.clean_final_seconds = args.clean_final_seconds
        
        self.ori_stride = self.stride
        self.ori_length = self.pose_length
        self.alignment = [0,0] # for beat


        self.ori_joint_list = joints_list[args.ori_joints]
        self.tar_joint_list = joints_list[args.tar_joints]
        
        ##----------------Copy from beat_sep_lower.py-----------##

        self.joints = len(list(self.tar_joint_list.keys()))
        self.joint_mask = np.zeros((len(list(self.ori_joint_list.keys()))+1)*3)
        for joint_name in self.tar_joint_list:
            if joint_name == "Hips":
                self.joint_mask[3:6] = 1
            else:
                self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
            
    #----------------Copy from beat_sep_lower.py-----------##
        
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
            # with open(f"{args.root_path}{args.train_data_path}vocab.pkl", 'rb') as f:
            with open(f"{args.root_path}/vocab.pkl", 'rb') as f:
                self.lang_model = pickle.load(f)


        # training multiple sub-modules in parallel at the same time
        
        # preloaded_dir = ""

        # if "mixamo_hand" in args.tar_joints:
        #     preloaded_dir = self.data_dir + f"{self.pose_rep}_hand_cache"
        # elif "mixamo_face" in args.tar_joints:
        #     preloaded_dir = self.data_dir + f"{self.pose_rep}_face_cache"
        # elif "mixamo_lower" in args.tar_joints:
        #     preloaded_dir = self.data_dir + f"{self.pose_rep}_lower_cache"
        # elif "mixamo_upper" in args.tar_joints:
        #      preloaded_dir = self.data_dir + f"{self.pose_rep}_upper_cache"
        # elif "mixamo_joint_full" in args.tar_joints: # using beat_skeleton_new
        #      preloaded_dir = self.data_dir + f"{self.pose_rep}_full_cache"

        if "skeleton_hand" in args.tar_joints:
            preloaded_dir = self.data_dir + self.args.cache_path + f"{self.pose_rep}_hand_cache"
        elif "skeleton_face" in args.tar_joints:
            preloaded_dir = self.data_dir + self.args.cache_path + f"{self.pose_rep}_face_cache"
        elif "skeleton_lower" in args.tar_joints:
            preloaded_dir = self.data_dir + self.args.cache_path + f"{self.pose_rep}_lower_cache"
        elif "skeleton_upper" in args.tar_joints:
             preloaded_dir = self.data_dir + self.args.cache_path + f"{self.pose_rep}_upper_cache"
        elif "skeleton_joint_full" in args.tar_joints: # using beat_skeleton_new
             preloaded_dir = self.data_dir + self.args.cache_path + f"{self.pose_rep}_full_cache"
            
        ##----------------------cả beat_sep_lower.py và beat_sep.py đều không dùng đoạn này-------------##
        ##beat.py 2022 thì có dùng
        # self.mean_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/bvh_mean.npy")
        # self.std_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/bvh_std.npy")
        self.audio_norm = args.audio_norm
        self.facial_norm = args.facial_norm
        if self.audio_norm:
            self.mean_audio = np.load(args.root_path+args.mean_pose_path+f"{args.audio_rep}/npy_mean.npy")
            self.std_audio = np.load(args.root_path+args.mean_pose_path+f"{args.audio_rep}/npy_std.npy")
        if self.facial_norm:
            self.mean_facial = np.load(args.root_path+args.mean_pose_path+f"{args.facial_rep}/json_mean.npy")
            self.std_facial = np.load(args.root_path+args.mean_pose_path+f"{args.facial_rep}/json_std.npy")

        # if self.args.beat_align:
        #     if not os.path.exists(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy"):
        #         self.calculate_mean_velocity(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
        #     self.avg_vel = np.load(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
        ##---------------------------------------------------------------------------------------------------##
            
        if build_cache and self.rank == 0:
            self.build_cache(preloaded_dir)
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"] 
            

            
    def build_cache(self, preloaded_dir):
        logger.info(f"Audio bit rate: {self.audio_fps}")
        logger.info("Reading data '{}'...".format(self.data_dir))
        
        # pose_length_extended = int(round(self.pose_length))
        logger.info("Creating the dataset cache...")

        if not os.path.exists(preloaded_dir):
            os.makedirs(preloaded_dir, exist_ok=True)
            
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
        
    
    def __len__(self):
        return self.n_samples
        
    ##--------This function copy from beat_sep_lower.py-----------##
    def idmapping(self, id):
        # map 1,2,3,4,5, 6,7,9,10,11,  12,13,15,16,17,  18,20,21,22,23,  24,25,27,28,30 to 0-24
        if id == 30: id = 8
        if id == 29: id = 0
        if id == 28: id = 14
        if id == 27: id = 19
        if id == 26: id = 0
        return id - 1

    ## Note: it's only need to mapping if there is missing file in the dataset
    ##------------------------------------------------------------##
    
    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds,  clean_final_seconds, is_test=False):
        self.n_out_samples = 0
        pose_files = sorted(glob.glob(os.path.join(self.data_dir, f"{self.pose_rep}") + "/*.bvh"), key=str,)  
        # create db for samples

        ##******************Note: Change desired FPS at self.pose_fps/xx*******************
        map_size = int(1024 * 1024 * 2048 * (self.audio_fps/16000)**3 * 4) * (len(pose_files)/30*(self.pose_fps/30)) * len(self.multi_length_training) * self.multi_length_training[-1] * 2 # in 1024 MB
        dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        n_filtered_out = defaultdict(int)
    
        for pose_file in pose_files:
            ext = ".npz" if "smplx" in self.args.pose_rep else ".bvh"
            pose_each_file = []
            pose_each_file_new = []
            trans_each_file = []
            audio_each_file = []
            facial_each_file = []
            word_each_file = []
            emo_each_file = []
            sem_each_file = []
            vid_each_file = []
            
            id_pose = pose_file.split("/")[-1][:-4] #1_wayne_0_1_1
            logger.info(colored(f"# ---- Building cache for Pose   {id_pose} ---- #", "blue"))

            pose_frames = []
            stride = int(30 / self.args.pose_fps)
            with open(pose_file, "r") as pose_data_file:
                for j, line in enumerate(pose_data_file.readlines()):
                    if j % stride != 0:
                        continue
                    frame_data = np.fromstring(line, dtype=float, sep=" ")
                    pose_frames.append(frame_data)
            
            pose_frames = np.array(pose_frames)  # (num_frames, (all_joint+1)*3)
            trans_each_file = pose_frames[:, :3]
            pose_frames = np.deg2rad(pose_frames)

            
            pose_frames = pose_frames * self.joint_mask # joint_mask không lấy position của hips
            rot_data = pose_frames[:, self.joint_mask.astype(bool)] # (num_frames, selected_joint*3)
            # print("data after select joint", rot_data.shape)
            

            
            pose_each_file = np.array(rot_data)
            # print("pose_each_file:", pose_each_file.shape)
            trans_each_file = np.array(trans_each_file)
            shape_each_file = np.repeat(np.array(-1).reshape(1, 1), pose_each_file.shape[0], axis=0)

            ## Detect contact for foot 4,5,6,10,11,12
            foot_joints_part1 = pose_each_file[:, 12:21]   # joints 5 to 7 (3 joints * 3 coords)
            # print("foot_joints_part1", foot_joints_part1.shape)
            foot_joints_part2 = pose_each_file[:, 30:39]  # joints 10 to 12 (3 joints * 3 coords)
            # print("foot_joints_part2", foot_joints_part2.shape)
            foot_joints = torch.tensor(np.concatenate([foot_joints_part1, foot_joints_part2], axis=1))
            foot_joints = foot_joints.reshape(rot_data.shape[0], 6, 3)
            # print("foot joints:", foot_joints.shape)
            feetv = torch.zeros(foot_joints.shape[1], foot_joints.shape[0])
            foot_joints = foot_joints.permute(1, 0, 2)
            # print("joints after permute:", foot_joints.shape)
            feetv[:, :-1] = (foot_joints[:, 1:] - foot_joints[:, :-1]).norm(dim=-1)
            # print("feettv:", feetv.shape)
            contacts = (feetv < 0.01).numpy().astype(float)
            # print("contacts:", contacts.shape)
            contacts = contacts.transpose(1, 0)
            
            pose_each_file = np.concatenate([pose_each_file, contacts], axis=1)
            # print("pose each file final:", pose_each_file.shape)

            
            if self.args.audio_rep is not None:
                logger.info(f"# ---- Building cache for Audio  {id_pose} and Pose {id_pose} ---- #")
                # print(f"{pose_file}")
                audio_file = pose_file.replace(self.args.pose_rep, 'wave16k').replace(ext, ".wav")
                print(f"audio file: {audio_file}")
                if not os.path.exists(audio_file):
                    logger.warning(f"# ---- file not found for Audio  {id_pose}, skip all files with the same id ---- #")
                    pose_files = [file for file in pose_files if file != id_pose]
                    # pose_files = pose_files.drop(pose_files[pose_files['id'] == id_pose].index)
                    # self.selected_file = self.selected_file.drop(self.selected_file[self.selected_file['id'] == id_pose].index) 
                    # selected_file la dataframe vi no dang su dung itterows() trong for loop, trong khi pose_files dang la kieu list
                    
                    continue
                audio_each_file, sr = librosa.load(audio_file)
                # print(f"audio_each_file load: {audio_each_file.shape}")
                audio_each_file = librosa.resample(audio_each_file, orig_sr=sr, target_sr=self.args.audio_sr)
                # print(f"audio_each_file resample: {audio_each_file.shape}")
                if self.args.audio_rep == "onset+amplitude":
                    from numpy.lib import stride_tricks
                    frame_length = 1024
                    # hop_length = 512
                    shape = (audio_each_file.shape[-1] - frame_length + 1, frame_length)
                    strides = (audio_each_file.strides[-1], audio_each_file.strides[-1])
                    rolling_view = stride_tricks.as_strided(audio_each_file, shape=shape, strides=strides)
                    amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
                    # pad the last frame_length-1 samples
                    amplitude_envelope = np.pad(amplitude_envelope, (0, frame_length-1), mode='constant', constant_values=amplitude_envelope[-1])
                    audio_onset_f = librosa.onset.onset_detect(y=audio_each_file, sr=self.args.audio_sr, units='frames')
                    onset_array = np.zeros(len(audio_each_file), dtype=float)
                    onset_array[audio_onset_f] = 1.0
                    # print(amplitude_envelope.shape, audio_each_file.shape, onset_array.shape)
                    audio_each_file = np.concatenate([amplitude_envelope.reshape(-1, 1), onset_array.reshape(-1, 1)], axis=1)
                elif self.args.audio_rep == "mfcc":
                    audio_each_file = librosa.feature.melspectrogram(y=audio_each_file, sr=self.args.audio_sr, n_mels=128, hop_length=int(self.args.audio_sr/self.args.audio_fps))
                    audio_each_file = audio_each_file.transpose(1, 0)
                    # print(audio_each_file.shape, pose_each_file.shape)
                if self.args.audio_norm and self.args.audio_rep == "wave16k": 
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
            # if self.id_rep is not None:
            #     vid_each_file.append(int(id_pose.split("_")[0])-1)
            if self.args.id_rep is not None:
                int_value = self.idmapping(int(id_pose.split("_")[0]))
                print("id:", int_value)
                vid_each_file = np.repeat(np.array(int_value).reshape(1, 1), pose_each_file.shape[0], axis=0)
                # print("vid_each_file:", vid_each_file)

            ## Note: it's only need to mapping if there is missing file in the dataset
                ## REMOVE ABOVE LINE BY THESE LINE IF NEED MAPPING
                # int_value = self.idmapping(int(f_name.split("_")[0]))
                # vid_each_file = np.repeat(np.array(int_value).reshape(1, 1), pose_each_file.shape[0], axis=0)
            
            ##------------------------------##
            
            # filtered_result = self._sample_from_clip(
            #     dst_lmdb_env,
            #     audio_each_file, pose_each_file, facial_each_file, word_each_file,
            #     vid_each_file, emo_each_file, sem_each_file,
            #     disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
            #     ) 

            filtered_result = self._sample_from_clip(
                dst_lmdb_env,
                audio_each_file, pose_each_file, trans_each_file, shape_each_file, facial_each_file, word_each_file,
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
            logger.info(colored("no. of excluded samples: {} ({:.1f}%)".format(
                n_total_filtered, 100 * n_total_filtered / (txn.stat()["entries"] + n_total_filtered)), "cyan"))
        dst_lmdb_env.sync()
        dst_lmdb_env.close()
    
    def _sample_from_clip(
        self, dst_lmdb_env, audio_each_file, pose_each_file, trans_each_file, shape_each_file, facial_each_file, word_each_file,
        vid_each_file, emo_each_file, sem_each_file,
        disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
        ):
        """
        for data cleaning, we ignore the data for first and final n s
        for test, we return all data 
        """
        # audio_start = int(self.alignment[0] * self.args.audio_fps)
        # pose_start = int(self.alignment[1] * self.args.pose_fps)
        #logger.info(f"before: {audio_each_file.shape} {pose_each_file.shape}")
        # audio_each_file = audio_each_file[audio_start:]
        # pose_each_file = pose_each_file[pose_start:]
        # trans_each_file = 
        #logger.info(f"after alignment: {audio_each_file.shape} {pose_each_file.shape}")
        #print(pose_each_file.shape)
        round_seconds_skeleton = pose_each_file.shape[0] // self.args.pose_fps  # assume 1500 frames / 15 fps = 100 s
        #print(round_seconds_skeleton)
        if audio_each_file != []:
            if self.args.audio_rep != "wave16k":
                round_seconds_audio = len(audio_each_file) // self.args.audio_fps # assume 16,000,00 / 16,000 = 100 s
            elif self.args.audio_rep == "mfcc":
                round_seconds_audio = audio_each_file.shape[0] // self.args.audio_fps
            else:
                round_seconds_audio = audio_each_file.shape[0] // self.args.audio_sr
            if facial_each_file != []:
                round_seconds_facial = facial_each_file.shape[0] // self.args.pose_fps
                logger.info(f"audio: {round_seconds_audio}s, pose: {round_seconds_skeleton}s, facial: {round_seconds_facial}s")
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
                max_round = max(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
                if round_seconds_skeleton != max_round: 
                    logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")  
            else:
                logger.info(f"pose: {round_seconds_skeleton}s, audio: {round_seconds_audio}s")
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton)
                max_round = max(round_seconds_audio, round_seconds_skeleton)
                if round_seconds_skeleton != max_round: 
                    logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")
        
        clip_s_t, clip_e_t = clean_first_seconds, round_seconds_skeleton - clean_final_seconds # assume [10, 90]s
        clip_s_f_audio, clip_e_f_audio = self.args.audio_fps * clip_s_t, clip_e_t * self.args.audio_fps # [160,000,90*160,000]
        clip_s_f_pose, clip_e_f_pose = clip_s_t * self.args.pose_fps, clip_e_t * self.args.pose_fps # [150,90*15]
        
        
        for ratio in self.args.multi_length_training:
            if is_test:# stride = length for test
                cut_length = clip_e_f_pose - clip_s_f_pose
                self.args.stride = cut_length
                self.max_length = cut_length
            else:
                self.args.stride = int(ratio*self.ori_stride)
                cut_length = int(self.ori_length*ratio)
                
            num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / self.args.stride) + 1
            logger.info(f"pose from frame {clip_s_f_pose} to {clip_e_f_pose}, length {cut_length}")
            logger.info(f"{num_subdivision} clips is expected with stride {self.args.stride}")
            
            if audio_each_file != []:
                audio_short_length = math.floor(cut_length / self.args.pose_fps * self.args.audio_fps)
                """
                for audio sr = 16000, fps = 15, pose_length = 34, 
                audio short length = 36266.7 -> 36266
                this error is fine.
                """
                logger.info(f"audio from frame {clip_s_f_audio} to {clip_e_f_audio}, length {audio_short_length}")
             
            n_filtered_out = defaultdict(int)
            sample_pose_list = []
            sample_audio_list = []
            sample_facial_list = []
            sample_shape_list = []
            sample_word_list = []
            sample_emo_list = []
            sample_sem_list = []
            sample_vid_list = []
            sample_trans_list = []
           
            for i in range(num_subdivision): # cut into around 2s chip, (self npose)
                start_idx = clip_s_f_pose + i * self.args.stride
                fin_idx = start_idx + cut_length 
                sample_pose = pose_each_file[start_idx:fin_idx]
                sample_trans = trans_each_file[start_idx:fin_idx]
                sample_shape = shape_each_file[start_idx:fin_idx]
                # print(sample_pose.shape)
                if self.args.audio_rep is not None:
                    audio_start = clip_s_f_audio + math.floor(i * self.args.stride * self.args.audio_fps / self.args.pose_fps)
                    audio_end = audio_start + audio_short_length
                    sample_audio = audio_each_file[audio_start:audio_end]
                else:
                    sample_audio = np.array([-1])
                sample_facial = facial_each_file[start_idx:fin_idx] if self.args.facial_rep is not None else np.array([-1])
                sample_word = word_each_file[start_idx:fin_idx] if self.args.word_rep is not None else np.array([-1])
                # sample_emo = emo_each_file[start_idx:fin_idx] if self.args.emo_rep is not None else np.array([-1])
                # sample_sem = sem_each_file[start_idx:fin_idx] if self.args.sem_rep is not None else np.array([-1])
                sample_vid = vid_each_file[start_idx:fin_idx] if self.args.id_rep is not None else np.array([-1])
                
                if sample_pose.any() != None:
                    # filtering motion skeleton data
                    sample_pose, filtering_message = MotionPreprocessor(sample_pose).get()
                    is_correct_motion = (sample_pose != [])
                    if is_correct_motion or disable_filtering:
                        sample_pose_list.append(sample_pose)
                        sample_audio_list.append(sample_audio)
                        sample_facial_list.append(sample_facial)
                        sample_shape_list.append(sample_shape)
                        sample_word_list.append(sample_word)
                        sample_vid_list.append(sample_vid)
                        # sample_emo_list.append(sample_emo)
                        # sample_sem_list.append(sample_sem)
                        sample_trans_list.append(sample_trans)
                    else:
                        n_filtered_out[filtering_message] += 1

            if len(sample_pose_list) > 0:
                with dst_lmdb_env.begin(write=True) as txn:
                    # for pose, audio, facial, shape, word, vid, emo, sem, trans in zip(
                    for pose, audio, facial, shape, word, trans, vid in zip(
                        sample_pose_list,
                        sample_audio_list,
                        sample_facial_list,
                        sample_shape_list,
                        sample_word_list,
                        # sample_emo_list,
                        # sample_sem_list,
                        sample_trans_list,
                        sample_vid_list,):
                        k = "{:005}".format(self.n_out_samples).encode("ascii")
                        # v = [pose, audio, facial, shape, word, emo, sem, vid, trans]
                        v = [pose, audio, facial, shape, word, trans, vid]
                        v = pyarrow.serialize(v).to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1
        return n_filtered_out

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pyarrow.deserialize(sample)
            # tar_pose, in_audio, in_facial, in_shape, in_word, emo, sem, vid, trans = sample
            tar_pose, in_audio, in_facial, in_shape, in_word, trans, vid = sample
            #print(in_shape)
            # vid = torch.from_numpy(vid).int()
            # emo = torch.from_numpy(emo).int()
            # sem = torch.from_numpy(sem).float() 
            in_audio = torch.from_numpy(in_audio).float() 
            in_word = torch.from_numpy(in_word).float() if self.args.word_cache else torch.from_numpy(in_word).int() 
            if self.loader_type == "test":
                tar_pose = torch.from_numpy(tar_pose).float()
                trans = torch.from_numpy(trans).float()
                in_facial = torch.from_numpy(in_facial).float()
                vid = torch.from_numpy(vid).float()
                in_shape = torch.from_numpy(in_shape).float()
            else:
                in_shape = torch.from_numpy(in_shape).reshape((in_shape.shape[0], -1)).float()
                trans = torch.from_numpy(trans).reshape((trans.shape[0], -1)).float()
                vid = torch.from_numpy(vid).reshape((vid.shape[0], -1)).float()
                tar_pose = torch.from_numpy(tar_pose).reshape((tar_pose.shape[0], -1)).float()
                in_facial = torch.from_numpy(in_facial).reshape((in_facial.shape[0], -1)).float()
            # return {"pose":tar_pose, "audio":in_audio, "facial":in_facial, "beta": in_shape, "word":in_word, "id":vid, "emo":emo, "sem":sem, "trans":trans}
            return {"pose":tar_pose, "audio":in_audio, "facial":in_facial, "beta": in_shape, "word":in_word, "trans":trans, "id":vid}

         
class MotionPreprocessor:
    def __init__(self, skeletons):
        self.skeletons = skeletons
        self.filtering_message = "PASS"

    def get(self):
        assert (self.skeletons is not None)

        # filtering
        if self.skeletons != []:
            if self.check_pose_diff():
                self.skeletons = []
                self.filtering_message = "pose"
            # elif self.check_spine_angle():
            #     self.skeletons = []
            #     self.filtering_message = "spine angle"
            # elif self.check_static_motion():
            #     self.skeletons = []
            #     self.filtering_message = "motion"

        # if self.skeletons != []:
        #     self.skeletons = self.skeletons.tolist()
        #     for i, frame in enumerate(self.skeletons):
        #         assert not np.isnan(self.skeletons[i]).any()  # missing joints

        return self.skeletons, self.filtering_message

    def check_static_motion(self, verbose=True):
        def get_variance(skeleton, joint_idx):
            wrist_pos = skeleton[:, joint_idx]
            variance = np.sum(np.var(wrist_pos, axis=0))
            return variance

        left_arm_var = get_variance(self.skeletons, 6)
        right_arm_var = get_variance(self.skeletons, 9)

        th = 0.0014  # exclude 13110
        # th = 0.002  # exclude 16905
        if left_arm_var < th and right_arm_var < th:
            if verbose:
                print("skip - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return True
        else:
            if verbose:
                print("pass - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return False


    def check_pose_diff(self, verbose=False):
#         diff = np.abs(self.skeletons - self.mean_pose) # 186*1
#         diff = np.mean(diff)

#         # th = 0.017
#         th = 0.02 #0.02  # exclude 3594
#         if diff < th:
#             if verbose:
#                 print("skip - check_pose_diff {:.5f}".format(diff))
#             return True
# #         th = 3.5 #0.02  # exclude 3594
# #         if 3.5 < diff < 5:
# #             if verbose:
# #                 print("skip - check_pose_diff {:.5f}".format(diff))
# #             return True
#         else:
#             if verbose:
#                 print("pass - check_pose_diff {:.5f}".format(diff))
        return False


    def check_spine_angle(self, verbose=True):
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        angles = []
        for i in range(self.skeletons.shape[0]):
            spine_vec = self.skeletons[i, 1] - self.skeletons[i, 0]
            angle = angle_between(spine_vec, [0, -1, 0])
            angles.append(angle)

        if np.rad2deg(max(angles)) > 30 or np.rad2deg(np.mean(angles)) > 20:  # exclude 4495
        # if np.rad2deg(max(angles)) > 20:  # exclude 8270
            if verbose:
                print("skip - check_spine_angle {:.5f}, {:.5f}".format(max(angles), np.mean(angles)))
            return True
        else:
            if verbose:
                print("pass - check_spine_angle {:.5f}".format(max(angles)))
            return False