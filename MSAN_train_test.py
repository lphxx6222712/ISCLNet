import json
import argparse
import os
import os.path as osp
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchnet.meter import ClassErrorMeter
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn


from model import MSA_subnet, RGA_subnet
from trainer import Trainer
from data.dataset import datalist
from data import data_transforms as dt
from utils.logger import Logger
from utils import FocalLoss
from utils.utils import count_param





def main(args):
    if args.resume:
       logger = Logger('./logs/'+'fundus_model:'+args.model_fundus+'_'+'OCT_model:'+args.model_OCT+'_'+args.modal+'_cross_%d'%args.cross+'_'+args.loss+'_' +args.optimizer+ '_'+'lr=%f'%args.learning_rate+'_'+'batch_size:%d'%args.batch_size+'_'+'.log')
    else:
       logger = Logger('./logs/'+'fundus_model:'+args.model_fundus+'_'+'OCT_model:'+args.model_OCT+'_'+args.modal+'_cross_%d'%args.cross+'_'+args.loss+'_' +args.optimizer+ '_'+'lr=%f'%args.learning_rate+'_'+'batch_size:%d'%args.batch_size+'_'+'.log', True)


    logger.append(vars(args))

    if args.display:
        writer = SummaryWriter()
    else:
        writer = None

    gpus = args.gpu.split(',')

    info = json.load(open(osp.join(args.list_dir, 'fundus'+'_'+'info.json'), 'r'))
    normalize = dt.Normalize(mean=info['mean'], std=info['std'])
    # data transforms
    t_fundus = []
    if args.resize:
        t_fundus.append(dt.Resize(args.resize))
    if args.random_rotate > 0:
        t_fundus.append(dt.RandomRotate(args.random_rotate))
    if args.random_scale > 0:
        t_fundus.append(dt.RandomScale(args.random_scale))
    if args.crop_size:
        t_fundus.append(dt.RandomCrop(args.crop_size))
    t_fundus.extend([dt.RandomHorizontalFlip(),
              dt.ToTensor(),
              normalize])


    info = json.load(open(osp.join(args.list_dir, 'OCT'+'_'+'info.json'), 'r'))
    normalize = dt.Normalize(mean=info['mean'], std=info['std'])
    t_OCT = []
    if args.resize:
        t_OCT.append(dt.Resize(args.resize))
    if args.random_rotate > 0:
        t_OCT.append(dt.RandomRotate(args.random_rotate))
    if args.random_scale > 0:
        t_OCT.append(dt.RandomScale(args.random_scale))
    if args.crop_size:
        t_OCT.append(dt.RandomCrop(args.crop_size))
    t_OCT.extend([dt.RandomHorizontalFlip(),
              dt.ToTensor(),
              normalize])


    t_ROI = []
    if args.resize:
        t_ROI.append(dt.Resize(args.resize))
    if args.random_rotate > 0:
        t_ROI.append(dt.RandomRotate(args.random_rotate))
    if args.random_scale > 0:
        t_ROI.append(dt.RandomScale(args.random_scale))
    if args.crop_size:
        t_ROI.append(dt.RandomCrop(args.crop_size))
    t_ROI.extend([dt.RandomHorizontalFlip(),
              dt.ToTensor(),
              normalize])

    train_dataset = datalist(args.data_dir, args.phase, 'train', dt.Compose(t_OCT),dt.Compose(t_fundus),dt.Compose(t_ROI), list_dir=args.list_dir, cross=args.cross)
    val_dataset = datalist(args.data_dir, args.phase, 'val', dt.Compose(t_OCT),dt.Compose(t_fundus),dt.Compose(t_ROI), list_dir=args.list_dir, cross=args.cross)
    train_dataloaders = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.workers, pin_memory=True,
                                                   drop_last=True)
    val_dataloaders = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.workers, pin_memory=True,
                                                   drop_last=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    is_use_cuda = torch.cuda.is_available()
    cudnn.benchmark = True

    model_fundus = MSA_subnet()
    model_OCT = RGA_subnet()

    if is_use_cuda and 1 == len(gpus):
        model_fundus = model_fundus.cuda()
        model_OCT = model_OCT.cuda()


    param_fundus = count_param(model_fundus)
    print(model_fundus.modules())
    print('###################################')
    print('Model #%s# parameters: %.2f M' % (args.model_fundus, param_fundus / 1e6))

    param_OCT = count_param(model_OCT)
    print(model_OCT.modules())
    print('###################################')
    print('Model #%s# parameters: %.2f M' % (args.model_OCT, param_OCT / 1e6))

    loss_fn = FocalLoss(gamma=2)
    optimizer_fundus = optim.Adam(model_fundus.parameters(), lr=args.learning_rate)
    optimizer_OCT = optim.Adam(model_OCT.parameters(), lr=args.learning_rate)

    lr_schedule_fundus = lr_scheduler.MultiStepLR(optimizer_fundus, milestones=[30, 60], gamma=0.1)
    lr_schedule_OCT = lr_scheduler.MultiStepLR(optimizer_OCT, milestones=[30, 60], gamma=0.1)

    metric = [[ClassErrorMeter([1,9], True)],[ClassErrorMeter([1,9], True)],[ClassErrorMeter([1,9], True)]]
    start_epoch = 0
    num_epochs  = args.epochs
    if args.phase == 'train':
        if args.model_path !=None:
            checkpoint = torch.load(args.model_path)
            model_fundus.load_state_dict(checkpoint['state_dict_fundus'])
            model_OCT.load_state_dict(checkpoint['state_dict_OCT'])
        my_trainer = Trainer(args, model_fundus, model_OCT,  loss_fn, optimizer_fundus, optimizer_OCT, lr_schedule_fundus, lr_schedule_OCT, args.log_batch, is_use_cuda, train_dataloaders,
                            val_dataloaders, metric, start_epoch, num_epochs, args.debug, logger, writer)
        my_trainer.fit()
        logger.append('Optimize Done!')
    elif args.phase == 'val':
        if args.model_path:
            checkpoint = torch.load(args.model_path)
            model_fundus.load_state_dict(checkpoint['state_dict_fundus'])
            model_OCT.load_state_dict(checkpoint['state_dict_OCT'])
        my_valer = Trainer(args, model_fundus, model_OCT, loss_fn, optimizer_fundus, optimizer_OCT, lr_schedule_fundus, lr_schedule_OCT, args.log_batch, is_use_cuda, train_dataloaders,
                            val_dataloaders, metric, start_epoch, num_epochs, args.debug, logger, writer)
        my_valer._valid()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-g', '--gpu', default='0', type=str,
                        help='GPU ID Select')
    parser.add_argument('--batch_size', default=8,
                         type=int, help='model train batch size')
    parser.add_argument('--display', default=False, dest='display',
                        help='Use TensorboardX to Display')
    parser.add_argument('-d', '--data-dir', default='data')
    parser.add_argument('-l', '--list-dir', default='data',
                        help='List dir to look for train_images.txt etc. '  
                             'I  t is the same with --data-dir if not set.')
    parser.add_argument('--modal', default='multi', type=str, help='choice: OCT, fundus, multi')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--random-rotate', default=0, type=int)
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--resize', default=(300,300), type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--phase', default='train')
    parser.add_argument('--model_fundus', default='resnet18_with_position_attention')
    parser.add_argument('--model_OCT', default='ROI_guided_OCT')
    parser.add_argument('--loss', default='Focal', help='BCE, softMargin, logitsBCE')
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--debug', default=0, dest='debug',
                        help='trainer debug flag')
    parser.add_argument('--log-batch', default=200)
    parser.add_argument('--cross', default=1, type=int)
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--num-cls', default=1, type=int)
    parser.add_argument('--describe', default='')
    parser.add_argument('--learning-rate',default=0.001)
    args = parser.parse_args()

    main(args)