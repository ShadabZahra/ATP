import torch
import clip

def description_prompt(data):
    descriptions = torch.cat([clip.tokenize(c) for _, c in data.descriptions])
    return descriptions
