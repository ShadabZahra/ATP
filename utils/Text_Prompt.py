import torch
import clip

def text_prompt(data):
    classes =  torch.cat([clip.tokenize(c) for _, c in data.classes])
    return classes
