import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


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




        


def initial_classifier(model, input_units, hidden_units):
    #freeze model parameter so we dont back propagate error (saves a lot of time)
    for param in model.parameters():
        param.requires_grad = False 

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_units, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(0.5)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    # Replacing the pretrained classifier with the one above
    model.classifier = classifier
        
    return model

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
def safe_model(model, save_dir, train_data, optimizer, epochs):
       
    checkpoint = {'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx,
                  'opt_state': optimizer.state_dict,
                  'num_epochs': epochs}
    print("All done, checkpoint saved as: ", save_dir)
    return torch.save(checkpoint, save_dir)
            

def main():

    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--data_directory', default='flowers', type = str, help='Enter path to data')
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str, help='Enter model architecture')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001, help='Enter LR')
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=1024, help='Enter hidden units')
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=2,help='enter epcoh number')
    parser.add_argument('--GPU', dest="gpu", action="store", type=str, default= True,help='Choose GPU or CPU (true = gpu ON), default CPU')
    user_input = parser.parse_args()

    # Get Keyword Args for Training
    data_dir = user_input.data_directory
    save_dir = user_input.save_dir
    learning_rate = user_input.learning_rate
    hidden_units = user_input.hidden_units
    epochs = user_input.epochs  
    arch = user_input.arch
    gpu_state = user_input.gpu

    ## PRETRAINED MODEL from user imput
    pretrained_model = arch

    #Get USER SELECTED model atrributes
    model = getattr(models,pretrained_model)(pretrained=True)
    input_units = model.classifier[0].in_features
    
    #Use model & attach new classifier created with user inputs
    model = initial_classifier(model, input_units, hidden_units)
    
    #defining criterion & optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    #defining device GPU/CPU and variables needed for training function

    device = torch.device("cuda:0" if gpu_state == True else "cpu") 
    print_every = 30
    steps = 0
    
    print("\nInitializing training process")
    trained_model = training(epochs, print_every, model, trainloader, validloader,optimizer ,criterion ,device)

    print("\nTraining process is completed!!")
    
    test_valid_loss, test_accuracy = validate(trained_model, testloader, criterion, device)
    print("test loss: ",test_valid_loss, 'test accuracy: ', test_accuracy)
   
    safe_model(trained_model, save_dir, train_data, optimizer, epochs)
if __name__ == '__main__': main()  