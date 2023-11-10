# Add Library
import numpy as np
import os, random
import matplotlib
import matplotlib.pyplot as plt
import json
import argparse
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils import data
from PIL import Image
# Process image
def process(path):
    pil_image = Image.open(path)
    pil_image.resize((256,256))
    width, height = pil_image.size 
    new_width, new_height = 224, 224
    left = round((width - new_width)/2)
    top = round((height - new_height)/2)
    x_right = round(width - new_width) - left
    x_bottom = round(height - new_height) - top
    right = width - x_right
    bottom = height - x_bottom
    pil_image = pil_image.crop((left, top, right, bottom))
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2 , 0, 1))
    tensor = torch.from_numpy(np_image)
    tensor = tensor.type(torch.FloatTensor)
    return tensor    
# Predict image follow image path
def predict(image_path, model, topk, device, cat_to_name):
    image = process(image_path)
    image = image.unsqueeze(0)
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        ps = torch.exp(model(image))
    ps, top_classes = ps.topk(topk, dim=1)
    idx_to_flower = {v:cat_to_name[k] for k, v in model.class_to_idx.items()}
    predicted_flowers_list = [idx_to_flower[i] for i in top_classes.tolist()[0]]
    return ps.tolist()[0], predicted_flowers_list
#Check model follow path
def ckLoad(path):
    ck = torch.load(path)
    if ck['vgg_type'] == "vgg11":
        model = torchvision.models.vgg11(pretrained=True)
    elif ck['vgg_type'] == "vgg13":
        model = torchvision.models.vgg13(pretrained=True)
    elif ck['vgg_type'] == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
    elif ck['vgg_type'] == "vgg19":
        model = torchvision.models.vgg19(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = ck['classifier']
    model.load_state_dict(ck['state_dict'])
    model.class_to_idx = ck['class_to_idx']
    return model
# Print a prediction 
def print_predict(args):
    model = ckLoad(args.model_filepath)
    print("...................................")
    if args.gpu and torch.cuda.is_available():
        dev = '.......This is a cuda.......'
    if args.gpu and not(torch.cuda.is_available()):
        dev = 'This is a cpu'
        print("Let use CPU")
    else:
        dev = ''
    print("...................................")
    model = model.to(dev)
    with open(args.category_names_json_filepath, 'r') as f:
        cat_to_name = json.load(f)
    top_ps, top_classes = predict(args.image_filepath, model, args.top_k, dev, cat_to_name)

    for i in range(args.top_k):
          print("#{:<3}{: <25} Prob:{:.2f}%".format(i, top_classes[i], top_ps[i]*100))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', action="store", type=float,default=0.01)
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
    parser.add_argument('--epochs', action="store", default=3, type=int)
    parser.add_argument('--dropout', action="store", type=float, default=0.5)
    parser.add_argument('--gpu', action="store", default="gpu")
    args = parser.parse_args()
    print_predict(args)