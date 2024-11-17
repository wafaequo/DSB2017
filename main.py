import os
print("Current Working Directory:", os.getcwd())
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("New Working Directory:", os.getcwd())

import sys
sys.path.append(r"c:\users\20192032\onedrive - tu eindhoven\documents\github\dsb2017")
print(sys.path)

#%%

from preprocessing.full_prep import full_prep
from config_submit import config as config_submit

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc
from data_detector import DataBowl3Detector,collate
from data_classifier import DataBowl3Classifier

from utils import *
from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import pandas

#%%
datapath = config_submit['datapath']
prep_result_path = config_submit['preprocess_result_path']
skip_prep = config_submit['skip_preprocessing']
skip_detect = config_submit['skip_detect']

#%%

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    if not skip_prep:
        testsplit = full_prep(datapath,prep_result_path,
                              n_worker = config_submit['n_worker_preprocessing'],
                              use_existing=config_submit['use_exsiting_preprocessing'])
    else:
        testsplit = os.listdir(datapath)

#%%
# Load the nodules detection model
nodmodel = import_module(config_submit['detector_model'].split('.py')[0])
config1, nod_net, loss, get_pbb = nodmodel.get_model()

# Load the pre-trained model weights
checkpoint = torch.load(config_submit['detector_param'], map_location=torch.device('cpu'), weights_only=False)  # Ensures loading on CPU
nod_net.load_state_dict(checkpoint['state_dict'])

# Use CPU instead of GPU
nod_net = nod_net.to(torch.device('cpu'))

#%%
# Ensure bbox_result_path exists
bbox_result_path = r'C:\Users\20192032\ownCloud\CT images/bbox_result'
if not os.path.exists(bbox_result_path):
    os.mkdir(bbox_result_path)
#%%
testsplit = os.listdir(datapath)
# Check if detection should be skipped
if __name__ == '__main__':
    # Your configuration and initialization
    margin = 32
    sidelen = 144
    config1['datadir'] = prep_result_path

    # Initialize SplitComb
    split_comber = SplitComb(
        sidelen, config1['max_stride'], config1['stride'], margin, pad_value=config1['pad_value']
    )

    # Create the dataset
    dataset = DataBowl3Detector(
        testsplit, config1, phase='test', split_comber=split_comber
    )

    # Adjust DataLoader for CPU usage
    test_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate
    )

    # Modify test_detect to run on CPU
    test_detect(
        test_loader, nod_net, get_pbb, bbox_result_path, config1, n_gpu=0  # Ensure n_gpu is 0 for CPU
    )

#%%
img = np.load(r'C:\Users\20192032\ownCloud\CT images\prep_results\Thorax 1.25_1.25_clean.npy')
pbb = np.load(r'C:\Users\20192032\ownCloud\CT images\bbox_result\bbox_result_pbb.npy')

#%%
from layers import nms,iou
pbb = pbb[pbb[:,0]>-1]
pbb = nms(pbb,0.05)
box = pbb[0].astype('int')[1:]

#%%
import matplotlib.patches as patches
import matplotlib.pyplot as plt
ax = plt.subplot(1,1,1)
plt.imshow(img[0,box[0]],'gray')
plt.axis('off')
rect = patches.Rectangle((box[2]-box[3],box[1]-box[3]),box[3]*2,box[3]*2,linewidth=2,edgecolor='red',facecolor='none')
ax.add_patch(rect)