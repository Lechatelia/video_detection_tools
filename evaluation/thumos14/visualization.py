import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from collections import defaultdict
from skvideo.io import vread, vwrite
import matplotlib.pyplot as plt
import cv2
import random

FRAME_DIR = '/data/DataSets/THUMOS14/frame'
Width = 171
Height = 128

parser = argparse.ArgumentParser(description="log analysis.py")
parser.add_argument('--tmp_file', type=str, default='tmp.txt', help="test log file path")
parser.add_argument('--framerate', type=int, default=25, help="frame rate")
# parser.add_argument('--annotation_dir', type=str, default='Evaluation/annotation', help="path of annotation")
parser.add_argument('--annotation_dir', type=str, default='/data/DataSets/THUMOS14/annotation', help="path of annotation")
parser.add_argument('--annotation_split', type=str, default='test', help="dataset split")
parser.add_argument('--output_dir', type=str, default='output', help="dataset split")
args = parser.parse_args()


import warnings

warnings.simplefilter("always")

class VideoWriter:
    def __init__(self, name, width, height, fps=25):
        # type: (str, int, int, int) -> None
        if not name.endswith('.avi') and not name.endswith('.mp4'):  # 保证文件名的后缀是.avi
            name += '.mp4'
            warnings.warn('video name should ends with ".avi or mp4"')
        self.__name = name          # 文件名
        self.__height = height      # 高
        self.__width = width
        # 宽
        if name.endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 如果是avi视频，编码需要为MJPG
        elif name.endswith('.mp4'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.__writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

    def write(self, frame):
        if frame.dtype != np.uint8:  # 检查frame的类型
            raise ValueError('frame.dtype should be np.uint8')
        # 检查frame的大小
        row, col, _ = frame.shape
        if row != self.__height or col != self.__width:
            warnings.warn('the width and height of input image is not the setting of this video')
            return
        self.__writer.write(frame)

    def close(self):
        self.__writer.release()


def read_pred(file):
    with open(file, 'r') as f:
        lines = f.read().splitlines()
        segments = {}
        for line in lines:
            seg = line.split()
            if seg[0] not in segments:
                segments[seg[0]] = []
            # segments[seg[0]].append({'label': int(seg[3]), 'score': float(seg[4]), 'segment': [float(seg[1]), float(seg[2])]})
            segments[seg[0]].append([float(seg[1]), float(seg[2]), float(seg[3]),float(seg[4])])
        for vid in segments:
            segments[vid].sort(key=lambda x: x[0])

        return segments


def dataset_label_parser(meta_dir, split, use_ambiguous=False):
    #meta_dir 存放label txt的路径
  class_id = defaultdict(int)
  id_class = {}
  # 读取class 标签
  with open(os.path.join(meta_dir, 'detclasslist.txt'), 'r') as f:
  # with open(os.path.join(meta_dir, 'Class Index.txt'), 'r') as f:
    lines = f.readlines()
    for l in lines:
      cname = l.strip().split()[-1]
      cid = int(l.strip().split()[0])
      class_id[cname] = cid
      id_class[cid] = cname
      if use_ambiguous:
        class_id['Ambiguous'] = 21
        id_class[21] = 'Ambiguous'
    segment = {} # 字典 存储每个video 里面的action 片段
    #内容是一个大的列表，具体内容是 [start_time. end_time. action_id]
    #video_instance = set()
  for cname in class_id.keys():
      # 读取相关类别的label文件 实际上是因为label是按照action类别分开放的
    tmp = '{}_{}.txt'.format(cname, split)
    with open(os.path.join(meta_dir, tmp)) as f:
      lines = f.readlines()
      for l in lines:
        vid_name = l.strip().split()[0] #video name
        start_t = float(l.strip().split()[1]) # start time
        end_t = float(l.strip().split()[2]) # end time
        #video_instance.add(vid_name)
        # initionalize at the first time
        #将结果内容放在这个列表当中
        if not vid_name in segment.keys():
          segment[vid_name] = [[start_t, end_t, class_id[cname]]]
        else:
          segment[vid_name].append([start_t, end_t, class_id[cname]])
#将每个video当中的action标签按照start_time顺序排序
  # sort segments by start_time
  for vid in segment:
      segment[vid].sort(key=lambda x: x[0])

  return id_class, segment

def show_segments(annotations,segments,id_class, split  ):
    Count_all = 10
    count_num = 0
    for video, pred in segments.items():
        # count_num = count_num + 1
        # 读取video
        print('video name :' + video + '.mp4')
        video_path = os.path.join(FRAME_DIR.replace('frame', 'video'), split,video+'.mp4')
        video_capture = cv2.VideoCapture(video_path)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print('orignal fps'+str(fps))

        # frame_path = os.path.join(FRAME_DIR, split,video)
        # images_dir_list = os.listdir(frame_path)
        # images_dir_list.sort(key=lambda x: x[6:11])
        # length =len(images_dir_list)
        #写入的视频流

        # for img_dir in images_dir_list:
        #
        #     image_list.append(img)
        #     video_steam.write(img)
        image_list = []
        #按帧去读取
        # 贴上标签
        # gt
        if video not in annotations:
            print('video' + video + 'not in annotations')
            continue
        labels = np.array(annotations[video])
        labels[:, :2] = np.round(labels[:, :2] * fps)
        labels = labels.astype(np.int)  # [num_segs, 3]
        #detection
        pres = np.array(segments[video])
        scores = pres[:, 3]
        pres[:, :2] = np.round(pres[:, :2] * fps)
        pres = pres[:,:3].astype(np.int)

        color_labels = np.array([[random.randint(0, 255) for _ in range(3)] for _ in range(len(labels))])
        color_pre = np.array([[random.randint(0, 255) for _ in range(3)] for _ in range(len(pres))])
        video_steam = VideoWriter(os.path.join(args.output_dir,video + '.avi'), width=width, height=height, fps=fps)

        for i in range(count):
            ret, img = video_capture.read()
            #plot gt
            seg_index = np.logical_and(labels[:,0]<=i,labels[:,1]>=i)
            if sum(seg_index)>0:
                det_cls = labels[seg_index, 2]
                plot_one_detections(img, [id_class[i] for i in det_cls], left=True, colors= color_labels[seg_index])

            #plot detection
            det_index = np.logical_and(pres[:,0]<=i,pres[:,1]>=i)
            if sum(det_index)>0:
                det_score = scores[det_index]
                # indices = np.argsort(det_score)[::-1]
                # det_index = det_index[indices]
                det_cls = pres[det_index, 2]
                # det_score = det_score[indices]
                plot_one_detections(img, [id_class[i] for i in det_cls], scores= det_score, left=False, colors=color_pre[det_index])

            video_steam.write(img)
        video_steam.close()
        print(video + '.avi  ' + 'OK')

        if count_num > Count_all:
            break

def plot_one_detections( img, labels, colors, scores = None, left = True,  line_thickness=None):  # Plots one bounding box on image img
    tl = line_thickness or round(0.0015 * max(img.shape[0:2])) + 1  # line thickness
    colors = colors.astype(np.int)
    if scores is not None:
        indices = np.argsort(scores)[::-1]
        labels = [labels[i] for i in indices]
        colors = colors[indices]
        scores = scores[indices]
    if left==True:
        x_coor = img.shape[1]*0.1
    else:
        x_coor = img.shape[1] * 0.6
    for i, label in enumerate(labels):
        if scores is not None:
            label = '%s %.2f'%(label, scores[i])
        color = [int(j)for j in colors[i]]
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c1 = (int(x_coor), int(i*t_size[1] +i + 3))
        c2 = (int(c1[0] + t_size[0]), int(c1[1] + t_size[1]))
        cv2.rectangle(img, c1, c2, tuple(color), -1)  # filled
        cv2.putText(img, label, (c1[0], c2[1]), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)




if __name__ == "__main__":
    segments = read_pred(args.tmp_file)
    # 如果use_ambiguous= true 是有213的目标video存在检测结果的
    # 如果false的话，只存在212的目标video
    #训练时实际上没有将ambiguous类包括在其中的
    id_class, annotations = dataset_label_parser(os.path.join(args.annotation_dir,'annotation_'+args.annotation_split),
                                      args.annotation_split, use_ambiguous=False)
    show_segments(annotations, segments, id_class, args.annotation_split)
