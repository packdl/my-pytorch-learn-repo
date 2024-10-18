from pathlib import Path

from dataloader import WeirdDataset
from torch.utils.data import DataLoader
from model_seq2seq import Seq2Seq
import torch
import torch.nn as nn


def cnt_model_params(model):
    """Count model parameters"""
    count = 0
    with torch.no_grad():
        for param in model.parameters():
            count+=param.numel()
    return count

def display_model_info(model_name, model):
    """ Display model information"""
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Module):
            count+=1
    print(model)
    print(f"{model_name}. parameters: {cnt_model_params(model)}")

if __name__=='__main__':
    """ loader = DataLoader(WeirdDataset('testing_label.json', 'testing_data'), batch_size=2)
    
    s = set()
    for X,y in loader:
        file_name, data = X
        print(type(file_name), type(data))
        
        for x in file_name:
            s.add(x)
        print(file_name)

        print(y.shape)

    print(len(s)) """

    model=Seq2Seq(4096, 128)
    model=model.to('cuda')

    if (Path('.')/'model1.pth').exists():
        print("Loading model parameters.")
        checkpoint = torch.load(Path('.')/'model1.pth', weights_only=True)
        model.load_state_dict(checkpoint)
        print("Model loaded")
        
    display_model_info('model1',model)
        


