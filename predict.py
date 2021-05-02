import argparse
import json
import PIL
import torch
from collections import OrderedDict
import numpy as np
from math import ceil
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image


    

def load_checkpoint(model, save_dir, gpu_state):
    
    
    if gpu_state == True:
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(save_dir, map_location=lambda storage, loc: storage)
    
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Converting image to PIL image using image file path
    pil_im = Image.open(f'{image}' + '.jpg')

    # Building image transform
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    
    ## Transforming image for use with network
    pil_tfd = transform(pil_im)
    
    # Converting to Numpy array 
    array_im_tfd = np.array(pil_tfd)
    
    # Converting to torch tensor from Numpy array
    img_tensor = torch.from_numpy(array_im_tfd).type(torch.FloatTensor)
    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img_tensor.unsqueeze_(0)
    
    return img_add_dim
def display(names, probs):   

    for i, j in enumerate(zip(probs, names)):
        print ("Rank {}:".format(i+1),
               "name: {}, probability: {}%".format(j[1], ceil(j[0]*100)))

def predict(image_tensor, model, cat_to_name, topk, gpu_state):
    # Predict the class (or classes) of an image using a trained deep learning model.
  
    # Setting model to evaluation mode and turning off gradients
    model.eval()

    if gpu_state == True:
        model.to('cuda')
    else:
        model.cpu()


    with torch.no_grad():
        output = model.forward(image_tensor)
    names = []
    # Calculating probabilities
    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    # Loading index and class mapping
    class_to_idx = model.class_to_idx
    # Inverting index-class dictionary
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    # Converting index list to class list
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]

    #Fill names with classes from top classes list
    for i in classes_top_list:
        names += [cat_to_name[i]]
    return probs_top_list, classes_top_list, names    


def main():
    ## PARSER
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image',dest='image_path',type=str, default='flowers/test/2/image_05100',help='Point to impage file for prediction.',)
    parser.add_argument('--checkpoint',dest="checkpoint",default='./checkpoint.pth', action="store",type=str,help='Point to checkpoint file as str.')
    parser.add_argument('--top_k',dest='top_k',type=int,default = 5,help='Choose top K matches as int.')
    parser.add_argument('--pretrearch', dest="pretrearch", action="store", default="vgg16", type = str, help='Enter model architecture')
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--GPU', dest="gpu", action="store", type=str, default= True,help='Choose GPU or CPU (true = gpu ON), default CPU')

    user_input_predict = parser.parse_args()
    

    # VARIABLES OUT OF PARSER
    checkpoint = user_input_predict.checkpoint
    image = user_input_predict.image_path
    top_k = user_input_predict.top_k
    gpu_state = user_input_predict.gpu
    cat_names = user_input_predict.category_names
    pretrained_model = user_input_predict.pretrearch


    with open(cat_names, 'r') as f:
        	cat_to_name = json.load(f)

    model = getattr(models,pretrained_model)(pretrained=True)        
    ## LOAD MODEL SAVED IN TRAIN.PY
    model = load_checkpoint(model,checkpoint,gpu_state)
    
    ## PROCESS IMAGE
    image_tensor = process_image(image)
    
    if gpu_state == True:
        image_tensor = image_tensor.to('cuda')
    else:
        pass

    ## PREDICT 
    probs, labels, names = predict(image_tensor, model, cat_to_name, top_k, gpu_state)
    
    display(names, probs)

if __name__ == '__main__': main()