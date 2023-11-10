#Import library
from collections import OrderedDict
import argparse
import torch
from torch import optim
import torch.nn as nn
from torchvision import transforms, datasets, models


# This function to buil model from model list
def do_build_model(archi='vgg19', hidden_units=2960, lr=0.001):
    models_list = {'vgg19': models.vgg19(pretrained=True),
                   'vgg11': models.vgg11(pretrained=True),
                   'vgg13': models.vgg13(pretrained=True),
                   'vgg16': models.vgg16(pretrained=True)}

    model = models_list[archi]
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units,  bias=True)),
        ('Relu1', nn.ReLU()),
        ('Dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 102,  bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    return model, criterion, optimizer
# Thí function to load data from folder
def data_from_folder(data_direc):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize, ])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.RandomCrop(224),
                                           transforms.ToTensor(),
                                           normalize, ])
    direc_train = data_direc + '/train'
    direc_valid = data_direc + '/valid'
    direct_test = data_direc + '/test'
    train_set = datasets.ImageFolder(direc_train, transform=train_transforms)
    valid_set = datasets.ImageFolder(direc_valid, transform=valid_transforms)
    test_set = datasets.ImageFolder(direct_test, transform=valid_transforms)
    return train_set, valid_set, test_set

#This function to validate data
def validate_data(model, testloader, criterion, dev='cuda'):
    test_loss = 0
    accuracy = 0
    for inputs, labels in testloader:
        inputs, labels = inputs.to(dev), labels.to(dev)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy
#this function to train
def do_train_model(model, trainset, validset, validation, epochs=8,
                dev='cuda'):
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=64)
    running_loss = 0
    for e in range(epochs):
        model.train()
        model.to(dev)
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validate_data(
                        model, validloader, criterion, dev)
                    print("{}/{}.. ".format(e + 1, epochs),
                          "Fail: {:.3f}.. ".format(running_loss / 100),
                          "Loss: {:.3f}.. ".format(test_loss / len(validloader)),
                          "Accuracy: {:.3f}".format(accuracy / len(validloader)))
                running_loss = 0
                model.train()
    model.class_to_idx = trainset.class_to_idx

    return model

def checkpoint_model(model, arch='vgg19', hidden_units=2960):
    checkpoint = {'arch': arch,
                  'hidden_units': hidden_units,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
                  }
    return checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('data_dir', action="store", default="./flowers/")
    parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
    parser.add_argument('--arch', action="store", default="vgg16")
    parser.add_argument('--learning_rate', action="store", type=float,default=0.01)
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
    parser.add_argument('--epochs', action="store", default=3, type=int)
    parser.add_argument('--dropout', action="store", type=float, default=0.5)
    parser.add_argument('--gpu', action="store", default="gpu")
    args = parser.parse_args()
    arch = args.arch if args.arch else 'vgg19'
    hidden_units = args.hidden_units if args.hidden_units else 2960
    lr = args.learning_rate if args.learning_rate else 0.001
    epochs = args.epochs if args.epochs else 8
    device = 'cuda' if args.gpu else 'cpu'

    trainset, validset, testset = data_from_folder(args.data)
    model, criterion, optimizer = do_build_model(
        arch=arch, hidden_units=hidden_units, lr=lr)
    model = do_train_model(model, trainset, validset,
                        validate_data, epochs, device=device)
    checkpoint = checkpoint_model(model, arch=arch, hidden_units=hidden_units)
    if args.save_dir:
        torch.save(checkpoint, args.save_dir + 'ck.pth')
    else:
        torch.save(checkpoint, 'ck.pth')
