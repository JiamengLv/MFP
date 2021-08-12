from __future__ import print_function    #超前使用python3的print函数
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from itertools import chain
from utils.dataset import DATASET
from utils.ImagePool import ImagePool
from model.Generator_new import Generator
import torch.nn.functional as F
from utils.fitsFun import DATASET_fits
from utils.fitsFun import LoadSaveFits
from utils.utils import *

#CUDA_VISIBLE_DEVICES=1 python my_script.py    参数
parser = argparse.ArgumentParser(description='train pix2pix model')
parser.add_argument('--batchSize', type=int, default=5, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--epoch', type=int, default=0, help='start epoch')
parser.add_argument('--n_epoch', type=int, default=10000, help='number of iterations to train for')
parser.add_argument('--lr', type=float, default= 0.02, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight decay in network D, default=1e-4')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='checkpoints/M_sun/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--train_data',default='./data/train_data/',help='image data')
parser.add_argument('--test_data',default='./data/test_data/',help='image data')
parser.add_argument('--fineSize', type=int, default=128, help='crop image to this size')
parser.add_argument('--flip', type=int, default=1, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--input_nc', type=int, default=1, help='channel number of input image')
parser.add_argument('--output_nc', type=int, default=2, help='channel number of output image')
parser.add_argument('--kernelSize', type=int, default=3, help='random crop kernel to this size')
parser.add_argument('--G_AB', default='', help='path to pre-trained G_AB')
parser.add_argument('--G_BA', default='', help='path to pre-trained G_BA')
parser.add_argument('--save_step', type=int, default=5, help='save interval')
parser.add_argument('--log_step', type=int, default=5, help='log interval')
parser.add_argument('--loss_type', default='mse', help='GAN loss type, bce|mse default is negative likelihood loss')
parser.add_argument('--poolSize', type=int, default=50, help='size of buffer in lsGAN, poolSize=0 indicates not using history')
parser.add_argument("--gpus", default="1", type=str, help="gpu ids (default: 0)")
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)  #用于递归创建目录
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    print("=> use  gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

opt.seed = random.randint(1, 10000)
print("Random Seed: ", opt.seed)
torch.manual_seed(opt.seed)
cudnn.benchmark = True

##########      dataset fits  #############
datasetA = DATASET_fits(opt.train_data,opt.fineSize)
loader_A= torch.utils.data.DataLoader(dataset=datasetA,batch_size=opt.batchSize,shuffle=True)
loaderA = iter(loader_A)

datasetB = DATASET_fits(opt.test_data,opt.fineSize)
loader_B= torch.utils.data.DataLoader(dataset=datasetB,batch_size=opt.batchSize,shuffle=True)
loaderB = iter(loader_B)

ABPool = ImagePool(opt.poolSize)
BAPool = ImagePool(opt.poolSize)

############   MODEL   ###########
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

ndf = opt.ndf
ngf = opt.ngf

G_AB = Generator(opt.input_nc, opt.output_nc, opt.ngf)
G_BA = Generator(opt.output_nc, opt.input_nc, opt.ngf)

if(opt.G_AB != ''):
    print('Warning! Loading pre-trained weights.')
    G_AB.load_state_dict(torch.load(opt.G_AB))
    G_BA.load_state_dict(torch.load(opt.G_BA))
else:
    G_AB.apply(weights_init)
    G_BA.apply(weights_init)
    
if(opt.cuda):
    G_AB.cuda()
    G_BA.cuda()

###########   LOSS & OPTIMIZER   ##########
criterionMSE = nn.L1Loss()
if(opt.loss_type == 'bce'):
    criterion = nn.BCELoss()
else:
    criterion = nn.MSELoss()
# chain is used to update two generators simultaneously
optimizerG = torch.optim.Adam(chain(G_BA.parameters()),lr=opt.lr, betas=(opt.beta1, 0.999))

############   GLOBAL VARIABLES   ###########
input_nc = opt.input_nc
output_nc = opt.output_nc
fineSize = opt.fineSize
batchSize = opt.batchSize
kernelSize = opt.kernelSize
real_A = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
label_A = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
AB = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
real_B = torch.FloatTensor(opt.batchSize, output_nc, fineSize, fineSize)
label_B = torch.FloatTensor(opt.batchSize, output_nc, fineSize, fineSize)
BA = torch.FloatTensor(opt.batchSize, output_nc, fineSize, fineSize)

real_A = Variable(real_A)
label_A = Variable(label_A)
real_B = Variable(real_B)
label_B = Variable(label_B)
AB = Variable(AB)
BA = Variable(BA)

if(opt.cuda):

    real_A = real_A.cuda()
    label_A = label_A.cuda()
    real_B = real_B.cuda()
    label_B = label_B.cuda()
    AB = AB.cuda()
    BA = BA.cuda()
    criterion.cuda()
    criterionMSE.cuda()


lossdata1 = []
lossdata = []
###########   Testing    ###########
def test(epoch):
    loaderB = iter(loader_B)
    imgB,label = loaderB.next()

    real_A.resize_(imgA[:,0:1,:,:].size()).copy_(imgA[:,0:1,:,:])
    real_B.resize_(imgA[:,1:3,:,:].size()).copy_(abs(imgA[:,1:3,:,:]))

    label_A.resize_(label[:, 0:1, :, :].size()).copy_(label[:, 0:1, :, :])
    label_B.resize_(label[:, 1:3, :, :].size()).copy_(abs(label[:, 1:3 , :, :]))

    BA = G_BA(real_B)
    AB = G_AB(real_A)

    # ABA = G_BA(AB)
    # BAB = G_AB(BA)

    # identity loss
    l_idt1 = (criterionMSE(AB, label_B) )
    l_idt2 =  (criterionMSE(BA, label_A))
    l_idt = l_idt2

    # reconstruction loss
    # l_rec_ABA = criterionMSE(ABA, real_A)
    # l_rec_BAB = criterionMSE(BAB, real_B)
    # errMSE = l_rec_ABA + l_rec_BAB

    # errG =  errMSE  + l_idt

    errG = l_idt

    h, w = 128, 128
    I = np.zeros(shape=(3, h, w))
    Abs_Br = np.zeros(shape=(3, h, w))
    Bt = np.zeros(shape=(3, h, w))

    I[0] = real_A.data.cpu().numpy()[0, :, :, :]
    I[1] = label_A.data.cpu().numpy()[0, :, :, :]
    I[2] = BA.data.cpu().numpy()[0, :, :, :]
    Abs_Br[0] = real_B.data.cpu().numpy()[0, 0:1, :, :]
    Abs_Br[1] = label_B.data.cpu().numpy()[0, 0:1, :, :]
    Abs_Br[2] = AB.data.cpu().numpy()[0, 0:1, :, :]
    Bt[0] = real_B.data.cpu().numpy()[0, 1:2, :, :]
    Bt[1] = label_B.data.cpu().numpy()[0, 1:2, :, :]
    Bt[2] = AB.data.cpu().numpy()[0, 1:2, :, :]

    LoadSaveFits.save_fit_cpu(I.data,'I_%03d_'%epoch,'./out_picture/M_sun/out_image_test/')
    LoadSaveFits.save_fit_cpu(Abs_Br.data,'Abs_Br_%03d_'%epoch,'./out_picture/M_sun/out_image_test/')
    LoadSaveFits.save_fit_cpu(Bt.data, 'Bt_%03d_' % epoch, './out_picture/M_sun/out_image_test/')
    ###########   Logging   ############
    if(epoch % opt.log_step):
        print('[%d/%d] Loss_errG: %.4f  Loss_idt: %.4f  '
                  % (epoch, opt.epoch,
                       errG.data,l_idt.data))
    lossdata.append('[%d] Loss_errG: %.4f  Loss_idt: %.4f  '
                  % (epoch, errG.data,l_idt.data))

    filepath = './mlosstest.txt'
    with open(filepath, 'w') as f:
            f.write(str(lossdata))



#损失图
# logger = Logger(opt.epoch, len(loader_A))

###########   Training   ###########
G_AB.train()
G_BA.train()

for epoch in range(opt.epoch,opt.n_epoch+1):
    ###########   data  ###########
    while True:

        try:
            imgA,label = loaderA.next()
        except StopIteration:
            loaderA = iter(loader_A)
            break

        #用来修改通道数，决定训练数据
        real_A.resize_(imgA[:,0:1,:,:].size()).copy_(imgA[:,0:1,:,:])
        real_B.resize_(imgA[:,1:3,:,:].size()).copy_(imgA[:,1:3,:,:])

        label_A.resize_(label[:,0:1,:,:].size()).copy_(label[:,0:1,:,:])
        label_B.resize_(label[:, 1:3, :, :].size()).copy_(abs(label[:, 1:3 , :, :]))
    
        ########### fGx ###########
        G_BA.zero_grad()
        G_AB.zero_grad()
    
        BA = G_BA(real_B)
        AB = G_AB(real_A)

        # ABA = G_BA(AB)
        # BAB = G_AB(BA)

        # identity loss
        l_idt = criterionMSE(BA,label_A)

        # reconstruction loss
        # l_rec_ABA = criterionMSE(ABA, real_A)
        # l_rec_BAB = criterionMSE(BAB, real_B)
        # errMSE = l_rec_ABA + l_rec_BAB

        # errG =  errMSE  + l_idt

        errG = l_idt
        errG.backward()
        optimizerG.step()

  
    ###########   Logging   ############
    if(epoch % opt.log_step):
        print('[%d/%d] Loss_errG: %.4f  Loss_idt1: %.4f  Loss_idt2: %.4f '
                  % (epoch, opt.epoch,
                       errG.data,criterionMSE(AB,label_B).data,criterionMSE(BA,label_A).data))
    lossdata1.append('[%d/%d] Loss_errG: %.4f  Loss_idt1: %.4f  Loss_idt2: %.4f '
                  % (epoch, opt.epoch,
                       errG.data,criterionMSE(AB,label_B).data,criterionMSE(BA,label_A).data))
    
    ########## Visualize #########
    if(epoch % opt.log_step == 0):

        h, w = 128, 128
        I = np.zeros(shape=(3, h, w))
        Abs_Br = np.zeros(shape=(3, h, w))
        Bt = np.zeros(shape=(3, h, w))

        I[0] = real_A.data.cpu().numpy()[0, :, :, :]
        I[1] = label_A.data.cpu().numpy()[0, :, :, :]
        I[2] = BA.data.cpu().numpy()[0, :, :, :]
        Abs_Br[0] = real_B.data.cpu().numpy()[0, 0:1, :, :]
        Abs_Br[1] = label_B.data.cpu().numpy()[0, 0:1, :, :]
        Abs_Br[2] = AB.data.cpu().numpy()[0, 0:1, :, :]
        Bt[0] = real_B.data.cpu().numpy()[0, 1:2, :, :]
        Bt[1] = label_B.data.cpu().numpy()[0, 1:2, :, :]
        Bt[2] = AB.data.cpu().numpy()[0, 1:2, :, :]

        LoadSaveFits.save_fit_cpu(I.data,'I_%03d_'%epoch,'./out_picture/M_sun/out_image_train/')
        LoadSaveFits.save_fit_cpu(Abs_Br.data,'Abs_Br_%03d_'%epoch,'./out_picture/M_sun/out_image_train/')
        LoadSaveFits.save_fit_cpu(Bt.data, 'Bt_%03d_' % epoch, './out_picture/M_sun/out_image_train/')
 
    if epoch % opt.save_step == 0:

        test(epoch)
        torch.save(G_BA.state_dict(), '{}/G_BA_{}.pth'.format(opt.outf, epoch))
        torch.save(G_AB.state_dict(), '{}/G_AB_{}.pth'.format(opt.outf, epoch))
        filepath = './mlosstrain.txt'
        with open(filepath, 'w') as f:
            f.write(str(lossdata1))


