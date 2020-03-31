# @Time : 2020/3/31 23:39 
# @Author : Jinguo Zhu
# @File : dataset_parser.py 
# @Software: PyCharm
'''
                    .::::.
                  .::::::::.
                 :::::::::::  I && YOU
             ..:::::::::::'
           '::::::::::::'
             .::::::::::
        '::::::::::::::..
             ..::::::::::::.
           ``::::::::::::::::
            ::::``:::::::::'        .:::.
           ::::'   ':::::'       .::::::::.
         .::::'      ::::     .:::::::'::::.
        .:::'       :::::  .:::::::::' ':::::.
       .::'        :::::.:::::::::'      ':::::.
      .::'         ::::::::::::::'         ``::::.
  ...:::           ::::::::::::'              ``::.
 ````':.          ':::::::::'                  ::::..
                    '.:::::'                    ':'````..
 '''
#因为thumos14 只有400多个视频真实有标签，这个程序是为了筛选那些有标签的视频

import os
import copy
import json
import pickle
import subprocess
import numpy as np
import cv2
import glob
import json
from collections import defaultdict

FRAME_DIR = 'evaluation/thumos14/Evaluation/annotation'
# FRAME_DIR = '/data/DataSets/THUMOS14/annotation'
META_DIR = os.path.join(FRAME_DIR, 'annotation_')

def dataset_label_parser(meta_dir, split, use_ambiguous=False, out_path='data/'):
    #meta_dir 存放label txt的路径
  class_id = defaultdict(int)
  # 读取class 标签
  cnt = 0
  with open(os.path.join(meta_dir, 'detclasslist.txt'), 'r') as f:
  # with open(os.path.join(meta_dir, 'Class Index.txt'), 'r') as f:
    lines = f.readlines()
    for l in lines:
      cname = l.strip().split()[-1]
      cid = int(l.strip().split()[0])
      class_id[cname] = cid
      if use_ambiguous:
        class_id['Ambiguous'] = 21
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
        cnt += 1
        if not vid_name in segment.keys():
          segment[vid_name] = [[start_t, end_t, class_id[cname]]]
        else:
          segment[vid_name].append([start_t, end_t, class_id[cname]])
#将每个video当中的action标签按照start_time顺序排序
  # sort segments by start_time
  for vid in segment:
    segment[vid].sort(key=lambda x: x[0])
    #将结果返回
  print("{} videos {} segments in the {} split.".format(len(segment.keys()), cnt, split))
  if True:
    keys = list(segment.keys())
    keys.sort()
    video_label = json.dumps(segment, indent=4)
    with open(os.path.join(out_path,'segment_{}.json'.format(split)), 'w') as f:
      f.write(video_label)

    # with open(os.path.join(out_path,'segment_{}.txt'.format(split)), 'w') as f:
    #   for k in keys:
    #     f.write("{}\n{}\n\n".format(k,segment[k]))
    #主要返回的是一个字典
    #字典的索引是video name
    #字典的内容是action片段标记，按照start time已经排序
  return segment

if __name__ == "__main__":
  # 注意这里面含有了一些ambiguius标签片段，也就是模糊不清的一些片段
  dataset_label_parser(META_DIR+ 'val', 'val', use_ambiguous=True)
  dataset_label_parser(META_DIR + 'test', 'test', use_ambiguous=True)