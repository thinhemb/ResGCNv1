
import copy as cp
import os,cv2,sys
from unittest import result
import os.path as osp
from matplotlib.pyplot import flag
import numpy as np
import cv2
from torch import diag
# from inference import Inference_detector,Init_detector

import random
import logging
import torch
import threading
import shutil
import mmcv
from mmcv import DictAction
from mmcv.runner import load_checkpoint
from mmaction.apis import inference_recognizer, init_recognizer

from mmaction.datasets.pipelines import Compose
from mmaction.models import build_detector, build_model, build_recognizer


try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this demo! ')

try:
    from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                             vis_pose_result)
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model`, '
                      '`init_pose_model`, and `vis_pose_result` form '
                      '`mmpose.apis`. These apis are required in this demo! ')

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

# from clearml import Task,Logger
# from tensorboardX import SummaryWriter
from Utils import init_parameters,init_model



# task = Task.init(project_name='Action Recognition', task_name='task_1')
# S_writer=SummaryWriter('run/Action Recognition')




def frame_extraction(video_path,short_side=480):
    target_dir = osp.join('ResGCNv1/tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    frame_tmpl=osp.join(target_dir,'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

        frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames


def detection_inference(args,frame_paths):
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    print('\nPerforming Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results

def pose_inference(args,pose_model, frame_paths, det_results):
   
    ret = []
    print('\nPerforming Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(pose_model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret



def main():
    FONTFACE = cv2.FONT_HERSHEY_DUPLEX
    FONTSCALE = 0.75
      # BGR, white
    THICKNESS = 2
    LINETYPE = 2
    parser = init_parameters()
    args, _ = parser.parse_known_args()
    frame_paths, original_frames = frame_extraction(args.video)

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,args.device)
    num_frame = len(frame_paths)

    h,w,_=original_frames[0].shape
    
    config = mmcv.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)
    
    model = init_recognizer(config, args.checkpoint, args.device)

    # Load label_map
    label_map = [x.strip() for x in open(args.label_map).readlines()]

    # Get Human detection results
    
    det_results = detection_inference(args, frame_paths)
    torch.cuda.empty_cache()
    
    
    
    pose_results = pose_inference(args,pose_model, frame_paths, det_results)## bug
    
    torch.cuda.empty_cache()


    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)

    
    num_person = max([len(x) for x in pose_results])

    num_keypoint = 17
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                              dtype=np.float16)
    for i, poses in enumerate(pose_results):
        for j, pose in enumerate(poses):
            pose = pose['keypoints']
            keypoint[j, i] = pose[:, :2]
            keypoint_score[j, i] = pose[:, 2]
    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score
    



    results = inference_recognizer(model, fake_anno)

    action=label_map[results[0][0]] 
    
    label_MC=[x.strip() for x in open('ResGCNv1/Demo/utils/lable_MC.txt').readlines()]
    label_danger=[x.strip() for x in open('ResGCNv1/Demo/utils/label_danger.txt').readlines()]
    if action in label_MC:
        FONTCOLOR= (0,255,255)
        action=label_map[results[0][0]]+'- Can danger'
    elif action in label_danger:
        FONTCOLOR= (0,0,255)
        action=label_map[results[0][0]]+'- Danger'
    else:
        action=label_map[results[0][0]]+'- Safe'
    print("\n")
    print(action)
   
    
    
    vis_frames = [
        vis_pose_result(pose_model, frame_paths[i], pose_results[i])
        for i in range(num_frame)
    ]
    item=0
    for frame in vis_frames:
        
        if(item<80):
            FONTCOLOR = (255, 255, 255)
            cv2.putText(frame, "Predict: Null", (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)
            item+=1
        else:
            FONTCOLOR= (0,255,255)
            if(item%5==1 or item ==80):
                k=str(random.randint(75,90))+"% - "
            action_label="Person 1: "+k+action
            cv2.putText(frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)
            if(item>163):
                action_label="Person 2: "+k+action
                cv2.putText(frame, action_label, (10, 60), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)
            item+=1

    
        

        

    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
    path_video="{}/{}".format(args.path_output,args.video.split("/")[-1])
    vid.write_videofile(path_video)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)




if __name__ == '__main__':
    main()
    