
import argparse
import json
import PIL
import torch
from collections import OrderedDict
import numpy as np
from math import ceil
from train import check_gpu
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image

def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    #parser.add_argument('--image',type=str,help='Point to impage file for prediction.',required=True)
    #parser.add_argument('--checkpoint',type=str,help='Point to checkpoint file as str.',required=True)
    parser.add_argument('--top_k',type=int,help='Choose top K matches as int.')
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

    args = parser.parse_args()
    
    return args

def load_checkpoint(checkpoint_path):
    
    checkpoint = torch.load("model_checkpoint.pth")
    
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    
    
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    

    im=Image.open(image)
    width=im.size[0]
    height=im.size[1]
    AspectRatio=width/height
    
    if width <= height:
        im=im.resize((256,int(256/AspectRatio)))
    else:
        im=im.resize((int(256*AspectRatio),256))
    
    midWidth=im.size[0]/2
    midHeight=im.size[1]/2
    cropped_im=im.crop((midWidth-112, midHeight-112, midWidth+112, midHeight+112))
    
    np_image=np.asarray(cropped_im)/255
    means=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    normalized_image=(np_image-means)/std
    final_image=normalized_image.transpose((2, 0, 1))
    
    return torch.from_numpy(final_image)
def display(flowers, probs):   

    for i, j in enumerate(zip(probs, flowers)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))

    
def predict(image_tensor, model, cat_to_name, top_k=5):

    model.eval();

    torch_image = torch.from_numpy(np.expand_dims(image_tensor, axis=0)).type(torch.FloatTensor)
   # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model=model.cpu()
   # model.to(device)
    #torch_image.to(device)
    
    with torch.no_grad():
        log_probs = model(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)
    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[i] for i in top_labels]
    top_flowers = [cat_to_name[x] for x in top_labels]
    
    return top_probs, top_labels, top_flowers

def main():
    
    args = arg_parser()
    
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)
    data_dir = 'flowers'
    test_dir = data_dir + '/test'
    model = load_checkpoint('model_checkpoint.pth')
    
   # image_tensor = process_image(args.image)
    #image_tensor = test_dir+'/2/image_05100.jpg'
    image_tensor = process_image(test_dir+'/2/image_05100.jpg')
    
    probs, labels, flowers = predict(image_tensor, model, cat_to_name)
    
    print(flowers, probs)
    display(flowers, probs)

if __name__ == '__main__': main()