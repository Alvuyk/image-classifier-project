import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--data_directory', default='flowers', type = str)
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args


## DATA transformation & Data loader  
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


transforms_training = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

transforms_validation = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

transforms_testing = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=transforms_training)
validate_data = datasets.ImageFolder(valid_dir, transform=transforms_validation)
test_data = datasets.ImageFolder(test_dir, transform=transforms_testing)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(validate_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)




##GPU checking
def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("NO GPU using CPU")
    return device


def pretrained_model(architecture="vgg16"):
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
        
    for param in model.parameters():
        param.requires_grad = False 
    return model

def initial_classifier(model, hidden_units):
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 1024)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.3)),
        ('fc4', nn.Linear(1024, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
        
    return classifier

def validate(model, validloader, criterion, device):
    valid_loss = 0
    valid_accuracy = 0
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return valid_loss/len(validloader), valid_accuracy/len(validloader) 

def training(epochs, print_every, model, trainloader, validloader, optimizer, criterion, device):
    model.to(device)
    steps = 0
    running_loss = 0
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss, valid_accuracy = validate(model, validloader, criterion, device) 

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss:.3f}.. "
                      f"Valid accuracy: {valid_accuracy:.3f}")
                running_loss = 0
                model.train()
                
    return model

# Function initial_checkpoint(Model, Save_Dir, Train_data) saves the model at a defined checkpoint
def safe_model(model, Save_Dir, Train_data, optimizer, epochs):
       
            model.class_to_idx = Train_data.class_to_idx
            torch.save({'structure' :'alexnet',
            'hidden_layer1':1024,
             'droupout':0.5,
             'epochs':1,
             'state_dict':model.state_dict(),
             'class_to_idx':model.class_to_idx,
             'optimizer_dict':optimizer.state_dict()},
             'checkpoint.pth')
            model.class_to_idx = Train_data.class_to_idx
            
            # Create checkpoint dictionary
            checkpoint = {'architecture': model.name,
                          'classifier': model.classifier,
                          'class_to_idx': model.class_to_idx,
                          'state_dict': model.state_dict()}
            
            # Save checkpoint
            torch.save(checkpoint, 'model_checkpoint.pth')
            print("All done, checkpoint saved")
def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    model = pretrained_model(architecture=args.arch)
    
    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)
    
    device = check_gpu(gpu_arg=args.gpu);
    model.to(device);
    
    
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print_every = 30
    steps = 0
    epochs = 1
    trained_model = training(args.epochs, print_every, model, trainloader, validloader,optimizer ,criterion ,device)
   # epochs, print_every, model, trainloader, validloader, optimizer, criterion, device
    print("\nTraining process is completed!!")
    
    validate(trained_model, testloader, criterion, device)
   
    safe_model(trained_model, args.save_dir, train_data, optimizer, args.epochs)
if __name__ == '__main__': main()  