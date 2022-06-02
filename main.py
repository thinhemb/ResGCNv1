

import os, yaml, argparse
from time import strftime

from src import utils as U
from src.generator import Generator
from src.processor import Processor
from src.visualizer import Visualizer


def main():
    # Loading Parameters
    parser = init_parameters()
    args, _ = parser.parse_known_args()

    # Updating Parameters (cmd > yaml > default)
    args = update_parameters(parser, args)

    # Setting save_dir
    save_dir = get_save_dir(args)
    U.set_logging(save_dir)
    with open('{}/config.yaml'.format(save_dir), 'w') as f:
        yaml.dump(vars(args), f)

    # Processing
    if args.generate_data or args.generate_label:
        g = Generator(args)
        g.start()

    elif args.extract or args.visualization:
        if args.extract:
            p = Processor(args, save_dir)
            p.extract()
        if args.visualization:
            v = Visualizer(args)
            v.start()

    else:
        p = Processor(args, save_dir)
        p.start()


def init_parameters():
    parser = argparse.ArgumentParser(description='Skeleton-based Action Recognition')

    # Setting
    parser.add_argument('--config', '-c', type=str, default='1012', help='ID of the using config', required=True)#ID của Sử dụng config
    parser.add_argument('--gpus', '-g', type=int, nargs='+', default=[], help='Using GPUs') #Sử dụng GPU
    parser.add_argument('--seed', '-s', type=int, default=1, help='Random seed') # thoi gian nghi
    parser.add_argument('--debug', '-db', default=False, action='store_true', help='Debug')
    parser.add_argument('--pretrained_path', '-pp', type=str, default='/home/thinh_do/Workplace/ResGCNv1/pretrained/1012_pa_resgcn_n51_r4_ntu_xset120.pth.tar', help='Path to pretrained models')#Đường dẫn đến các mô hình được đào tạo trước
    parser.add_argument('--work_dir', '-w', type=str, default='', help='Work dir')
    parser.add_argument('--no_progress_bar', '-np', default=False, action='store_true', help='Do not show progress bar') #Không hiển thị thanh tiến trình
    parser.add_argument('--path', '-p', type=str, default='', help='Path to save preprocessed skeleton files')#Đường dẫn để lưu các tệp bộ xương được xử lý trước

    # Processing
    parser.add_argument('--resume', '-r', default=False, action='store_true', help='Resume from checkpoint')#Tiếp tục từ trạm kiểm soát
    parser.add_argument('--evaluate', '-e', default=False, action='store_true', help='Evaluate')  #Đánh giá
    parser.add_argument('--extract', '-ex', default=False, action='store_true', help='Extract')#Trích xuất
    parser.add_argument('--visualization', '-v', default=False, action='store_true', help='Visualization') #Hình dung
    parser.add_argument('--generate_data', '-gd', default=True, action='store_true', help='Generate skeleton data')#Tạo dữ liệu bộ xương
    parser.add_argument('--generate_label', '-gl', default=True, action='store_true', help='Only generate label') #Chỉ tạo nhãn

    # Visualization
    parser.add_argument('--visualization_class', '-vc', type=int, default=0, help='Class: 1 ~ 60, 0 means true class')
    parser.add_argument('--visualization_sample', '-vs', type=int, default=0, help='Sample: 0 ~ batch_size-1')
    parser.add_argument('--visualization_frames', '-vf', type=int, nargs='+', default=[], help='Frame: 0 ~ max_frame-1')

    # Dataloader
    parser.add_argument('--dataset', '-d', type=str, default='', help='Select dataset')#Chọn tập dữ liệu
    parser.add_argument('--dataset_args', default=dict(), help='Args for creating dataset')#Args để tạo tập dữ liệu

    # Model
    parser.add_argument('--model_type', '-mt', type=str, default='', help='Model type')
    parser.add_argument('--model_args', default=dict(), help='Args for creating model')#Args để tạo tập dữ liệu

    # Optimizer
    parser.add_argument('--optimizer', '-o', type=str, default='', help='Initial optimizer')#Trình tối ưu hóa ban đầu
    parser.add_argument('--optimizer_args', default=dict(), help='Args for optimizer')

    # LR_Scheduler
    parser.add_argument('--lr_scheduler', '-ls', type=str, default='', help='Initial learning rate scheduler')#Trình lập lịch tốc độ học ban đầu
    parser.add_argument('--scheduler_args', default=dict(), help='Args for scheduler')#Args dành cho trình lập lịch

    return parser


def update_parameters(parser, args):
    if os.path.exists('./configs/{}.yaml'.format(args.config)):
        with open('./configs/{}.yaml'.format(args.config), 'r') as f:
            try:
                yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_arg = yaml.load(f)
            default_arg = vars(args)
            for k in yaml_arg.keys():
                if k not in default_arg.keys():
                    raise ValueError('Do NOT exist this parameter {}'.format(k))
            parser.set_defaults(**yaml_arg)
    else:
        raise ValueError('Do NOT exist this file in \'configs\' folder: {}.yaml!'.format(args.config))
    return parser.parse_args()


def get_save_dir(args):
    if args.debug or args.evaluate or args.extract or args.visualization or args.generate_data or args.generate_label:
        save_dir = '{}/temp'.format(args.work_dir)
    else:
        ct = strftime('%Y-%m-%d %H-%M-%S')
        save_dir = '{}/{}_{}_{}/{}'.format(args.work_dir, args.config, args.model_type, args.dataset, ct)
    U.create_folder(save_dir)
    return save_dir


if __name__ == '__main__':
    os.chdir(os.getcwd())
    main()
