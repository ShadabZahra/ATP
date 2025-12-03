from datasets import Action_DATASETS
from modules.Visual_Prompt import visual_prompt
from utils.Augmentation import get_augmentation
from utils.Text_Prompt import *
from utils.Description_Prompt import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from dotmap import DotMap
import torch.nn as nn
import numpy
import torch
import random
import os
import clip
import wandb
import argparse
import shutil
import yaml

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)
    
def validate(val_loader, classes, descriptions, device, model, fusion_model, config, num_classes):
    # Accuracy counters
    num = 0
    corr_1 = corr1_1 = corr1_3 = 0
    with torch.no_grad():
        text_features = model.encode_text(classes.to(device))
        description_features = model.encode_text(descriptions.to(device))
        for _, (image, class_id) in enumerate(tqdm(val_loader)):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            # Encode image
            image_input = image.to(device).view(-1, 3, h, w)
            frame_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(frame_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            flat_frame = frame_features.view(-1, frame_features.shape[-1])
            description_features /= description_features.norm(dim = -1, keepdim =True)
            text_features /= text_features.norm(dim = -1, keepdim =True)
            flat_frame /= flat_frame.norm(dim=-1, keepdim=True)

            score_class = (flat_frame @ text_features.T * 100).view(b,t, num_classes)
            score_class = score_class.mean(dim = 1)

            score_description = (flat_frame @ description_features.T * 100).view(b,t,num_classes)
            score_description = score_description.mean(dim = 1)

            
            class_scores = torch.maximum(score_description , score_class).softmax(dim=-1)
            _, top1_idxs = score_class.topk(1, dim=-1)
            _, top1_description = score_description.topk(1, dim = -1)
            _, indices_1 = class_scores.topk(1, dim=-1)
            num += b

            for i in range(b):
                if indices_1[i] == class_id[i]: corr_1 += 1
                if top1_idxs[i] == class_id[i]: corr1_1 += 1
                if top1_description[i] == class_id[i]: corr1_3 += 1

    top1 = (corr_1 / num) * 100
    acc1 = (corr1_1 / num) * 100
    top_description = (corr1_3 / num) * 100
    wandb.log({'ClassNameOnly': acc1})
    wandb.log({"DualAdaption": top1})
    wandb.log({ 'DescriptionOnly': top1_description})
    print('*' * 80)
    print(f'DualAdaption: {top1}')
    print(f'ClassNameOnly: {acc1}')
    print(f'DescriptionOnly, {top_description}')
    print('*' * 80)

    return top1


def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()
    

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    working_dir = os.path.join(
        './exp',
        config['network']['type'],
        config['network']['arch'],
        config['data']['dataset'],
        args.log_time
    )
    wandb.init(
        project=config['network']['type'],
        name='{}_{}_{}_{}'.format(
            args.log_time,
            config['network']['type'],
            config['network']['arch'],
            config['data']['dataset']
        ),
        mode="offline"  # logs only locally
    )

    config = DotMap(config)
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, clip_state_dict = clip.load(
        config.network.arch,
        device=device,
        jit=False,
        tsm=config.network.tsm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout
    )

    transform_val = get_augmentation(False, config)

    fusion_model = visual_prompt(
        config.network.sim_header,
        clip_state_dict,
        config.data.num_segments
    )
    model_text  = TextCLIP(model).to(device)
    model_image = ImageCLIP(model).to(device)
    fusion_model = fusion_model.to(device)

    if torch.cuda.is_available():
        model_text  = torch.nn.DataParallel(model_text)
        model_image = torch.nn.DataParallel(model_image)
        fusion_model = torch.nn.DataParallel(fusion_model)

    wandb.watch(model)
    wandb.watch(fusion_model)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(model_text)
        clip.model.convert_weights(model_image)

    num_classes = config.data.num_classes
    if os.path.isfile(config.pretrain):
        print(("=> loading checkpoint '{}'".format(config.pretrain)))
        checkpoint = torch.load(config.pretrain, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        if config.network.sim_header != 'meanP':
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
        del checkpoint

    val_lists = []
    for key in ["val_list1", "val_list2", "val_list3"]:
        if key in config.data and config.data[key]:
            val_lists.append(config.data[key])

    results = []
    root = config.data.root
    set_seed(config.seed)
    for idx, vlist in enumerate(val_lists, 1):
        print(f"\n===== Evaluating on Split {idx} ({vlist}) =====")

        val_data = Action_DATASETS(
            root,
            vlist,
            config.data.label_list,
            config.data.description_list,
            num_segments=config.data.num_segments,
            image_tmpl=config.data.image_tmpl,
            transform=transform_val,
            random_shift=config.random_shift
        )
        val_loader = DataLoader(
            val_data,
            batch_size=config.data.batch_size,
            num_workers=config.data.workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
        classes= text_prompt(val_data)
        descriptions= description_prompt(val_data)
        prec1 = validate(
            val_loader,
            classes,
            descriptions,
            device,
            model,
            fusion_model,
            config,
            num_classes
        )
        results.append(prec1)

    if results:
        mean_acc = sum(results) / len(results)
        std_acc = (sum((x - mean_acc) ** 2 for x in results) / len(results)) ** 0.5
        print(f"\nFinal Results across {len(results)} splits:")
        print(f"  Mean Acc: {mean_acc:.2f}%")
        print(f"  Std Dev : {std_acc:.2f}%")
        wandb.log({"val/mean_acc": mean_acc, "val/std_acc": std_acc})

if __name__ == '__main__':
    main()