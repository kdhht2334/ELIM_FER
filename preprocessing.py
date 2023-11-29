# -*- coding: utf-8 -*-
import os, shutil
from os import listdir
from os.path import isfile, join
from PIL import Image

import csv

import numpy as np
import scipy.misc as ms
import glob

import cv2
from tqdm import tqdm

import torch
from torchvision.utils import save_image


def save_im(tensor, title):
    image = tensor.cpu().clone()
    x = image.clamp(0, 255)/255.
    x = x.view(x.size(0), 224, 224)
    save_image(x, "{}".format(title))


if __name__ == "__main__":


    # ------------------------------------
    # Face detection using facenet-pytorch
    # (install) $ pip install facenet-pytorch
    # (reference) https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch
    # ------------------------------------
    
    root_dir = r'/<DATA_DIR>/aff_wild2/train_ext2/'
    
    from facenet_pytorch import MTCNN  #, InceptionResnetV1
    mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=False)  # keep_all=False (detect single face)
    
    pic_list = sorted([f for f in listdir(root_dir) if not isfile(join(root_dir, f))])
    
    for i in range(1):

        tmpdir = r'/<DATA_DIR>/aff_wild2/cropped_faces_train_ext1/'+pic_list[i]
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
    
        ll = []
        ll.append(sorted(glob.glob(root_dir + pic_list[i] + '/*.png'), key=os.path.getmtime))
        ll = ll[0]
        print("*-- Folder name is {}".format(pic_list[i]))
        
    
        for j in tqdm(range(len(ll))):
            
            if j == 0:
                img_cropped = torch.ones(size=(3,224,224))
                
            NAME = ll[j][:-4]
            img = Image.open(NAME+'.png')
        
            # Get cropped and prewhitened image tensor
            try:
                img_cropped = mtcnn(img, save_path=tmpdir+'/'+ll[j].split('/')[-1])
            except TypeError:
                save_im(img_cropped, tmpdir+'/'+ll[j].split('/')[-1])
            
                
    #----------------
    # Data annotation
    #----------------

    mypath = r'/<DATA_DIR>/aff_wild1/train_ext2/'
    ll = glob.glob('/<DATA_DIR>/aff_wild2/cropped_faces_train_ext1/*')
    ll.sort()
    
    ll1 = []
    for i in range(len(ll)):
        if len(ll[i].split('/')[-1].split('.')) == 1:
            ll1.append(ll[i].split('/')[-1])
    
    
    remove_idx = 2
    for j in range(len(ll1)):
    
        mylist = [['subDirectory_filePath', 'valence', 'arousal']]
        
        mypath = '/<DATA_DIR>/aff_wild2/cropped_faces_train_ext1/'+str(ll1[j])+'/*'
        img_path = sorted(glob.glob(mypath), key=os.path.getmtime)
        
        with open('/<DATA_DIR>/aff_wild2/Anno/Train_Set/'+str(ll1[j])+'.txt') as f:
            anno_path = f.read().splitlines()
        anno_path.pop(0)
        
        img_path = img_path[remove_idx-1::remove_idx]
        anno_path = anno_path[remove_idx-1::remove_idx]
            
        ### make csv file using list
        name_list = []
        for i in range(len(img_path)):
            nn = img_path[i].split('/')[-2] + '/' + img_path[i].split('/')[-1]
            name_list.append(nn)
            
        min_value = np.min([len(anno_path), len(name_list)])
        for i in range(min_value):
            mylist.append([name_list[i], 
                           anno_path[i].split(',')[0], 
                           anno_path[i].split(',')[1]])
        
        del img_path, anno_path
        
        
    with open('/<DATA_DIR>/cropped_faces_train_ext2/training.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(mylist)
