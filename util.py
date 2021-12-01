import torch
from matplotlib import pyplot as plt
import cmasher as cmr
import numpy as np
import matplotlib
from tqdm.auto import tqdm

def compute_g(net,x,target_idx=None):
    net.eval()
    x.requires_grad = True
    output = net(x)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    if not target_idx: target_idx = torch.argmax(probabilities, 1).item()
    index = torch.ones((probabilities.size()[0], 1)) * target_idx
    index = torch.tensor(index, dtype=torch.int64)
    loss = probabilities.gather(1, index)
    net.zero_grad()
    loss.backward(torch.ones_like(loss))
    x.requires_grad = False
    grad = x.grad.data
    return grad



# def coalition(x):
    
    
# def colalition_approximation(net,x,target_idx=None):
#     net.eval()
#     x.requires_grad = False
#     for 
#     output = net(x)
#     probabilities = torch.nn.functional.softmax(output, dim=1)
    
    
#     return grad

class IG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
def compute_ig(net,x,target_idx=None,m=3,baseline=None,batch_size=16):
    assert m>=3
    net.eval()
    output = net(x)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    if not target_idx: target_idx = torch.argmax(probabilities, 1).item()
    scaled_inputs = [baseline+alpha*(x-baseline) for alpha in np.linspace(0,1,m)]
    scaled_inputs = torch.cat(scaled_inputs)
    grads = torch.cat([compute_g(net, scaled_inputs[i:i+batch_size], target_idx=target_idx) for i in tqdm(range(0,len(scaled_inputs),batch_size),desc='ig')],axis=0)
    grads = (grads[:-1] + grads[1:]) / 2.0
    grads_avg = grads.mean(dim=0).reshape(x.shape)
    ig = (x-baseline)*grads_avg
    return ig

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
