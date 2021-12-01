import torch
from matplotlib import pyplot as plt
import cmasher as cmr
import numpy as np
import matplotlib
from tqdm.auto import tqdm

def normalize(A):
    A-=A.min()
    A/=A.max()
    return A

def proccess_grad(grad,min=0.0,max=1):
    grad = grad.mean(axis=2)
    grad = normalize(grad)
    grad = torch.clamp(grad,min=min,max=max)
    grad = normalize(grad)
    grad = torch.log(grad + 0.1)
    return grad

def vis_grad(grad,min=0.0,max=1,polarity='both'):
    neg_grad = grad.clone()
    pos_grad = grad.clone()
    neg_grad[neg_grad>0]=0
    neg_grad = -neg_grad
    pos_grad[pos_grad<0]=0

    neg_grad = proccess_grad(neg_grad,min=min,max=max)
    pos_grad = proccess_grad(pos_grad,min=min,max=max)
    
    cmap = plt.get_cmap('cmr.sunburst')
    norm = None
    if polarity=='+':
        grad = pos_grad
    elif polarity=='-':
        grad = - neg_grad
    elif polarity=='both':
        grad = pos_grad + neg_grad
    elif polarity=='':
        grad = pos_grad - neg_grad
        cmap = plt.get_cmap('cmr.iceburn')
        norm = matplotlib.colors.TwoSlopeNorm(vmin=grad.min(), vcenter=0, vmax=grad.max())
    return grad,cmap,norm
