from re import A
import cv2, logging, os,argparse,sys,yaml
import torch
import numpy as np
 
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)
from src import utils as U
from src.generator import Generator
from src.processor import Processor
from src.visualizer import Visualizer
import src.model as model
from src.dataset.graph import Graph






def init_parameters():
    parser = argparse.ArgumentParser(description='Skeleton-based Action Recognition')

    # Setting
    
    parser.add_argument('--gpus', '-g', type=int, nargs='+', default=[], help='Using GPUs') #Sử dụng GPU
    parser.add_argument('--seed', '-s', type=int, default=1, help='Random seed') # thoi gian nghi

    parser.add_argument('--type_test', type=int, default=0,help='0: video, 1: camera')
    parser.add_argument('--video', type=str, default='ResGCNv1/Demo/video_headache_1.mp4',help='path video test')

    parser.add_argument('--pretrained_path', '-pp', type=str, default='ResGCNv1/pretrained/1011_pa-resgcn-n51-r4_ntu-xsub120.pth.tar', help='Path to pretrained models')#Đường dẫn đến các mô hình được đào tạo trước
    parser.add_argument('--work_dir', '-w', type=str, default='', help='Work dir')
    
    parser.add_argument('--path_output', '-p', type=str, default='ResGCNv1/Demo/output', help='Path to save preprocessed skeleton files')#Đường dẫn để lưu các tệp bộ xương được xử lý trước
    parser.add_argument('--label_map',type=str,default='ResGCNv1/Demo/utils/label_map.txt',help='Path to label map file')
   
    
    parser.add_argument('--evaluate', '-e', default=False, action='store_true', help='Evaluate')  #Đánh giá
    parser.add_argument('--extract', '-ex', default=False, action='store_true', help='Extract')#Trích xuất
    
    # Dataloader
    parser.add_argument('--dataset', '-d', type=str, default='ntu-xsub120', help='Select dataset')#Chọn tập dữ liệu
    # parser.add_argument('--dataset_args', default=dict(), help='Args for creating dataset')#Args để tạo tập dữ liệu
    

    # Model
    parser.add_argument('--model_type', '-mt', type=str, default='resgcn-b19', help='Model type')
    parser.add_argument('--model_args', default=[9,2], help='Args for creating model')#Args để tạo tập dữ liệu
    #Config model
    parser.add_argument('--pose_config', type=str, default='ResGCNv1/Demo/utils/pose_config.py', help='Path to config file')
    parser.add_argument('--pose_checkpoint', default='ResGCNv1/Demo/utils/pose_checkpoint.pth', help='Args for config file')
    
    parser.add_argument('--det_config', type=str, default='ResGCNv1/Demo/utils/det_config.py', help='Path to config file')
    parser.add_argument('--det_checkpoint', default='ResGCNv1/Demo/utils/det_checkpoint.pth', help='Args for config file')
    
    parser.add_argument('--config', type=str, default='ResGCNv1/Demo/utils/skeletion_config.py', help='Path to config file')
    parser.add_argument('--checkpoint', default='ResGCNv1/Demo/utils/skeletion_checkpoint.pth', help='Args for config file')
    
    parser.add_argument('--cfg_options', default={}, help='Args for config file')
    
    parser.add_argument('--device', type=str, default='cpu', help='CPU/CUDA device option')
    parser.add_argument('--det_score_thr',type=float,default=0.9,help='det score threshold')
    
    
    return parser



def init_model(args):
    graph = Graph(args.dataset)
    
    kwargs = {
            'data_shape': [3,6,300,25,2],
            'num_class': 120,
            'A': torch.Tensor(graph.A),
            'parts': [torch.Tensor(part).long() for part in graph.parts]
    }
            
    
    output_device = None
    if output_device is not None:
        device =  torch.device('cuda:{}'.format(output_device))
    else:
        device =  torch.device('cpu')

    net=model.create(args.model_type,**kwargs).to(device)
    net=torch.nn.DataParallel(net,'cpu',output_device=output_device)
    
    logging.info('Model: {} {}'.format(args.model_type, args.model_args))
    logging.info('Model parameters: {:.2f}M'.format(
        sum(p.numel() for p in net.parameters()) / 1000 / 1000
    ))
    
    pretrained_model=args.pretrained_path
    if os.path.exists(pretrained_model):
        checkpoint = torch.load(pretrained_model, map_location=torch.device('cpu'))
        net=net.module.load_state_dict(checkpoint['model'])
        logging.info('Pretrained model: {}'.format(pretrained_model))
    elif args.pretrained_path:
        logging.warning('Warning: Do NOT exist this pretrained model: {}'.format(pretrained_model))
    
    return net



def read_yaml(filepath):
    ''' Input a string filepath, 
        output a `dict` containing the contents of the yaml file.
    '''
    with open(filepath, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded

