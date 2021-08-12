import PIL.Image as Image
from astropy.io import fits
import torchvision.transforms as transforms
import numpy as np
import torch
import os
import torch.utils.data as Data
from skimage import exposure
import random
class LoadSaveFits:
    def __init__(self,path,img,name):
        self.path = path
        self.img = img
        self.name = name
        
    def norm(img):
        img = (img - np.min(img))/(np.max(img) - np.min(img)) #normalization
        img -= np.mean(img)  # take the mean
        img /= np.std(img)  #standardization
        img = np.array(img,dtype='float32')
        return img
 
    def norm2(img,z):
        for i in range(z):
            img[i] = (img[i] - np.min(img[i]))/(np.max(img[i]) - np.min(img[i])) #normalization
            img[i] -= np.mean(img[i])  # take the mean
            img[i] /= np.std(img[i])  #standardization           #使得数据的标准差为1
            img[i] = np.array(img[i],dtype='float32')
        return img
       
    def read_fits(path):
        hdu = fits.open(path)
        img = hdu[0].data
        img = np.array(img,dtype = np.float32)
        hdu.close()
        return img
    
    def save_fit_cpu(img,name,path):
        if os.path.exists(path + name+'.fits'):
            os.remove(path + name+'.fits')
        grey=fits.PrimaryHDU(img)
        greyHDU=fits.HDUList([grey])
        greyHDU.writeto(path + name+'.fits')
        
    def save_fit(img,name,path):
        if torch.cuda.is_available(): 
            img = torch.Tensor.cpu(img)
            img = img.data.numpy()
            IMG = img[0,0,:,:]
        else:
            img = np.array(img)
        if os.path.exists(path + name+'.fits'):
            os.remove(path + name+'.fits')
        grey=fits.PrimaryHDU(IMG)
        greyHDU=fits.HDUList([grey])
        greyHDU.writeto(path + name+'.fits')

# load data of cycleCNN
class DATASET_fits():
    def __init__(self,dataPath='',fineSize=512):
        super(DATASET_fits, self).__init__()
        # list all images into a list
        self.list = os.listdir(dataPath)
        self.list.sort()
        #print(self.list)
        self.dataPath = dataPath
        self.fineSize = fineSize

        
    def __getitem__(self, index):

        path = os.path.join(self.dataPath,self.list[index])

        imgA = LoadSaveFits.read_fits(path)
        z,h,w = imgA.shape
        number_rot = random.randint(0,3)
        img1 = imgA[:,int((h/2-self.fineSize/2)):int((h/2+self.fineSize/2)),int((w/2-self.fineSize/2)):int((w/2+self.fineSize/2))]
        LoadSaveFits.save_fit_cpu(img1.data, '123' , './')
        z, h, w = img1.shape
        input_img = np.zeros(shape=(3,h,w))
        input_img[0] = img1[0]
        input_img[1] = abs(img1[1])
        input_img[2] = np.sqrt(img1[2]**2 + img1[3]**2 )
        input_img = LoadSaveFits.norm2(input_img,z-1)
        input_img = torch.from_numpy(input_img)

        label_img = np.zeros(shape=(3,h,w))
        label_img[0] = img1[0]
        label_img[1] = abs(img1[1])
        label_img[2] = np.sqrt(img1[2]**2 + img1[3]**2 )
        label_img = torch.from_numpy(label_img)
        return input_img,label_img
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.list)



        

    




    
