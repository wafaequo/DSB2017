import argparse
import os
import time
import numpy as np
from importlib import import_module
import shutil
from utils import *
import sys
from split_combine import SplitComb

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc

def test_detect(data_loader, net, get_pbb, save_dir, config,n_gpu):
    start_time = time.time()
    net.eval()
    split_comber = data_loader.dataset.split_comber
    
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        s = time.time()
        
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1]
        shortname = name.split('_clean')[0]
        data = data[0][0]
        coord = coord[0][0]
        
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        n_per_run = 1
        
        splitlist = [0, len(data) + 1]

# Calculate valid split indices
        splitlist = [0]
        batch_size = data.size(0)  # This is 12 in your case
        chunk_size = 4  # You can adjust this value
        # Create splits
        for i in range(chunk_size, batch_size, chunk_size):
            splitlist.append(i)

        splitlist.append(batch_size)  # Ensure we end with the full batch size

# Validate the splitlist
        print(f"Valid splitlist: {splitlist}")
        
        outputlist = []
        featurelist = []
        
        for i in range(len(splitlist)-1):
            input = Variable(data[splitlist[i]:splitlist[i+1]], requires_grad=False)
            inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]], requires_grad=False)
            
            with torch.no_grad():  # Disable gradient calculation for inference
                if isfeat:
                    output, feature = net(input, inputcoord)
                    featurelist.append(feature.data.cpu().numpy())  # Ensure feature is on CPU
                else:
                    output = net(input, inputcoord)
                    

            outputlist.append(output.data.cpu().numpy())
            
        output = np.concatenate(outputlist,0)

        output = split_comber.combine(output,nzhw=nzhw)

        if isfeat:
            feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])[:,:,:,:,:,np.newaxis]
            feature = split_comber.combine(feature,sidelen)[...,0]

        thresh = -3
        pbb,mask = get_pbb(output,thresh,ismask=True)
        
        if isfeat:
            feature_selected = feature[mask[0],mask[1],mask[2]]
            np.save(os.path.join(save_dir, shortname+'_feature.npy'), feature_selected)
            
        #tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
        #print([len(tp),len(fp),len(fn)])
        print([i_name,shortname])
        e = time.time()
        
        file_path_name = os.path.join(save_dir, r'C:\Users\20192032\ownCloud\CT images\bbox_result'+'_pbb.npy')
        np.save(file_path_name, pbb)
        #np.save(os.path.join(save_dir, shortname+'_lbb.npy'), lbb)
    end_time = time.time()


    print('elapsed time is %3.2f seconds' % (end_time - start_time))
    print
    print
