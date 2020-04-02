import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import sys
import io
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import argparse
import torchvision
from PIL import Image

import numpy as np
import json
from pytorch_i3d import InceptionI3d

import pdb

print('CUDA available: {}'.format(torch.cuda.is_available()))

print('the available CUDA number is : {}'.format(torch.cuda.device_count()))

def load_frame(frame_file, resize=False):

    data = Image.open(frame_file)

    # assert(data.size[1] == 256)
    # assert(data.size[0] == 340)

    if resize:
        data = data.resize((224, 224), Image.ANTIALIAS)
    else :
        data = data.resize((340, 256), Image.ANTIALIAS)

    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert(data.max()<=1.0)
    assert(data.min()>=-1.0)

    return data


def load_zipframe(zipdata, name, resize=False):

    stream = zipdata.read(name)
    data = Image.open(io.BytesIO(stream))

    assert(data.size[1] == 256)
    assert(data.size[0] == 340)

    if resize:
        data = data.resize((224, 224), Image.ANTIALIAS)

    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert(data.max()<=1.0)
    assert(data.min()>=-1.0)

    return data




def oversample_data(data): # (39, 16, 224, 224, 2)  # Check twice

    data_flip = np.array(data[:,:,:,::-1,:])

    data_1 = np.array(data[:, :, :224, :224, :])
    data_2 = np.array(data[:, :, :224, -224:, :])
    data_3 = np.array(data[:, :, 16:240, 58:282, :])   # ,:,16:240,58:282,:
    data_4 = np.array(data[:, :, -224:, :224, :])
    data_5 = np.array(data[:, :, -224:, -224:, :])

    data_f_1 = np.array(data_flip[:, :, :224, :224, :])
    data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
    data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
    data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
    data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

    return [data_1, data_2, data_3, data_4, data_5,
        data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]




def load_rgb_batch(frames_dir, rgb_files, 
                   frame_indices, resize=False):

    if resize:
        batch_data = np.zeros(frame_indices.shape + (224,224,3))
    else:
        batch_data = np.zeros(frame_indices.shape + (256,340,3))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):

            batch_data[i,j,:,:,:] = load_frame(os.path.join(frames_dir, 
                rgb_files[frame_indices[i][j]]), resize)

    return batch_data


def load_ziprgb_batch(rgb_zipdata, rgb_files, 
                   frame_indices, resize=False):

    if resize:
        batch_data = np.zeros(frame_indices.shape + (224,224,3))
    else:
        batch_data = np.zeros(frame_indices.shape + (256,340,3))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):

            batch_data[i,j,:,:,:] = load_zipframe(rgb_zipdata, 
                rgb_files[frame_indices[i][j]], resize)

    return batch_data


def load_flow_batch(frames_dir, flow_x_files, flow_y_files, 
                    frame_indices, resize=False):

    if resize:
        batch_data = np.zeros(frame_indices.shape + (224,224,2))
    else:
        batch_data = np.zeros(frame_indices.shape + (256,340,2))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):

            batch_data[i,j,:,:,0] = load_frame(os.path.join(frames_dir, 
                flow_x_files[frame_indices[i][j]]), resize)

            batch_data[i,j,:,:,1] = load_frame(os.path.join(frames_dir, 
                flow_y_files[frame_indices[i][j]]), resize)

    return batch_data


def load_zipflow_batch(flow_x_zipdata, flow_y_zipdata, 
                    flow_x_files, flow_y_files, 
                    frame_indices, resize=False):

    if resize:
        batch_data = np.zeros(frame_indices.shape + (224,224,2))
    else:
        batch_data = np.zeros(frame_indices.shape + (256,340,2))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):

            batch_data[i,j,:,:,0] = load_zipframe(flow_x_zipdata, 
                flow_x_files[frame_indices[i][j]], resize)

            batch_data[i,j,:,:,1] = load_zipframe(flow_y_zipdata, 
                flow_y_files[frame_indices[i][j]], resize)

    return batch_data



def run(mode='rgb', load_model='', split='', sample_mode='oversample', frequency=16,
    input_dir='', output_dir='', batch_size=40, usezip=False, segment_json=''):

    output_dir = os.path.join(output_dir, split)
    input_dir = os.path.join(input_dir, split)
    segment_json = segment_json.format(split)
    with open(segment_json, 'r') as json_file:
        json_video_label = json_file.read()
        # json.loads() load后面的s就是load str的意思，所以先要read成str， 再load str to json
        video_label = json.loads(json_video_label)
        video_names = list(video_label.keys())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    chunk_size = 16

    assert(mode in ['rgb', 'flow'])
    assert(sample_mode in ['oversample', 'center_crop', 'resize'])
    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    
    #i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d = torch.nn.DataParallel(i3d).cuda()


    i3d.eval()  # Set model to evaluate mode

    def forward_batch(b_data):
        b_data = b_data.transpose([0, 4, 1, 2, 3])    # 将channel提前
        b_data = torch.from_numpy(b_data)   # b,c,t,h,w  # bsx3x16x224x224
        with torch.no_grad():
            b_data =b_data.cuda().float()
            b_features = i3d(b_data, features=True) # [bs, 1024, 1, 1, 1]
        
        b_features = b_features.data.cpu().numpy()[:,:,0,0,0] # [bs, 1024]
        return b_features


    # video_names = [i for i in os.listdir(input_dir) if i[0] == 'v']

    for video_name in video_names: # 确保一下video list的数量是200/213

        save_file = '{}-{}.npz'.format(video_name, mode)
        if save_file in os.listdir(output_dir): #如果存在了就不在进行特征提取
            continue

        frames_dir = os.path.join(input_dir, video_name)


        if mode == 'rgb':
            if usezip:
                rgb_zipdata = zipfile.ZipFile(os.path.join(frames_dir, 'img.zip'), 'r')
                rgb_files = [i for i in rgb_zipdata.namelist() if i.startswith('img')]
            else:
                rgb_files = [i for i in os.listdir(frames_dir) if i.startswith('img')]

            rgb_files.sort()
            frame_cnt = len(rgb_files)

        else:
            if usezip:
                flow_x_zipdata = zipfile.ZipFile(os.path.join(frames_dir, 'flow_x.zip'), 'r')
                flow_x_files = [i for i in flow_x_zipdata.namelist() if i.startswith('x_')]

                flow_y_zipdata = zipfile.ZipFile(os.path.join(frames_dir, 'flow_y.zip'), 'r')
                flow_y_files = [i for i in flow_y_zipdata.namelist() if i.startswith('y_')]
            else:
                flow_x_files = [i for i in os.listdir(frames_dir) if i.startswith('flow_x')]
                flow_y_files = [i for i in os.listdir(frames_dir) if i.startswith('flow_y')]

            flow_x_files.sort()
            flow_y_files.sort()
            assert(len(flow_y_files) == len(flow_x_files))
            frame_cnt = len(flow_y_files)



        # clipped_length = (frame_cnt // chunk_size) * chunk_size   # Cut frames

        # Cut frames
        assert(frame_cnt > chunk_size)
        clipped_length = frame_cnt - chunk_size
        clipped_length = (clipped_length // frequency) * frequency  # The start of last chunk

        frame_indices = [] # Frames to chunks
        for i in range(clipped_length // frequency + 1):
            frame_indices.append(
                [j for j in range(i * frequency, i * frequency + chunk_size)])

        frame_indices = np.array(frame_indices) # [num_snippet, chunk_size] 代表着网络的输入

        #frame_indices = np.reshape(frame_indices, (-1, 16)) # Frames to chunks
        chunk_num = frame_indices.shape[0] # 最终的snippet数量

        batch_num = int(np.ceil(chunk_num / batch_size))    # Chunks to batches 要进行多少个batch 向上取整
        frame_indices = np.array_split(frame_indices, batch_num, axis=0)
        # 注意此处分到batch_num个batch中，为了避免最后一个batch太少，做了一个均分，这里的操作实际上相当于重新规定了batch——szie
        # 这里实际batch_size 会小于等于命令行中的那个batch——size
        # 比如9个样本，batch——size开始规定的是4，那么原本应该是4 4 1 进行特征提取
        # 这里batch——num=3.实际上的batch-size 就会变成 3，3，3


        if sample_mode == 'oversample':
            full_features = [[] for i in range(10)]
        else:
            full_features = [[]]

        for batch_id in range(batch_num):

            # 注意 我们提取的视频帧
            require_resize = sample_mode == 'resize'

            if mode == 'rgb':
                if usezip:
                    batch_data = load_ziprgb_batch(rgb_zipdata, rgb_files, 
                        frame_indices[batch_id], require_resize)
                else:                
                    batch_data = load_rgb_batch(frames_dir, rgb_files, 
                        frame_indices[batch_id], require_resize)
            else:
                if usezip:
                    batch_data = load_zipflow_batch(
                        flow_x_zipdata, flow_y_zipdata,
                        flow_x_files, flow_y_files, 
                        frame_indices[batch_id], require_resize)
                else:
                    batch_data = load_flow_batch(frames_dir, 
                        flow_x_files, flow_y_files, 
                        frame_indices[batch_id], require_resize)

            if sample_mode == 'oversample':
                batch_data_ten_crop = oversample_data(batch_data)

                for i in range(10):
                    pdb.set_trace()
                    assert(batch_data_ten_crop[i].shape[-2]==224)
                    assert(batch_data_ten_crop[i].shape[-3]==224)
                    full_features[i].append(forward_batch(batch_data_ten_crop[i]))

            else:
                if sample_mode == 'center_crop':
                    batch_data = batch_data[:,:,16:240,58:282,:] # Centrer Crop  (39, 16, 224, 224, 2)
                
                assert(batch_data.shape[-2]==224)
                assert(batch_data.shape[-3]==224)
                # batch_data [batch_szie, 16, 2224, 224, 3]
                full_features[0].append(forward_batch(batch_data))


        #full_features是一个 list 如果是多次采样的话，len(list)=10 一般 len(list)=1 元素是每一个batch的特征 [bs,1024]
        full_features = [np.concatenate(i, axis=0) for i in full_features] # 将每一个采样方式的特征拼接 元素为[num_snippet, 1024]
        full_features = [np.expand_dims(i, axis=0) for i in full_features] # 元素为[1,num_snippet, 1024]
        full_features = np.concatenate(full_features, axis=0) # [采样方式, num_snippet, 1024] 除了oversample，其他的采样方式都是1

        np.savez(os.path.join(output_dir, save_file), 
            feature=full_features,
            frame_cnt=frame_cnt,
            video_name=video_name)

        # 如果想要调用就是
        # np.load('output/rgb/val/video_validation_0000266-rgb.npz')['feature']

        print('{} done: {} / {}, {}'.format(
            video_name, frame_cnt, clipped_length, full_features.shape))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='rgb', type=str)
    parser.add_argument('--load_model', default='models/rgb_imagenet.pt',type=str)
    parser.add_argument('--input_dir', default='/data/DataSets/THUMOS14/frames',  type=str)
    parser.add_argument('--segment_json', default='data/segment_{}.json',  type=str)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--output_dir', default='output/rgb', type=str)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--sample_mode', default='resize', type=str)
    parser.add_argument('--frequency', type=int, default=16)
    # 注意默认是等于chunk_size=16 可以实际调整，当frequency小于16时，网络的输入就会有overlap，大于16时，就会有一些帧数据被舍弃
    parser.add_argument('--usezip', default=False, type=bool)
    parser.add_argument('--no-usezip', default=False,type=bool)
    parser.set_defaults(usezip=False)

    args = parser.parse_args()

    run(mode=args.mode, 
        load_model=args.load_model,
        sample_mode=args.sample_mode,
        input_dir=args.input_dir,
        split = args.split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        frequency=args.frequency,
        usezip=args.usezip,
        segment_json=args.segment_json)
