import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

# each folder contains 46 images, each folder represents a class
Train_folder = 'D:\\XXX\\Train'
Test_folder = 'D:\\XXX\\Test'

# list of classes (names of the folders)
classes = sorted(os.listdir(Train_folder))
epochs = 300
batch_size = 46

def load_dataset(data_path_train: str, data_path_test: str):
    """
    Data loader. Data transformations have been applied to the images.
    transforms.AutoAugment() shows the same increase in accuracy as commented transformations.
    """
    # Original train set of images was too small to provide enough images \
    # Data augmentation applied to solve the problem
    transformation_train_augmentation = transforms.Compose([
        transforms.Grayscale(),
        transforms.AutoAugment(),

        # transforms.RandomRotation((-30, 30)),
        # transforms.RandomHorizontalFlip(0.9),
        # transforms.RandomVerticalFlip(0.9),
        # transforms.Resize((128, 128)),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Transformations applied to the train set of images
    transformation_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Transformations applied to the test set of images
    transformation_test = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset_normal = torchvision.datasets.ImageFolder(
        root=data_path_train,
        transform=transformation_train
    )

    train_dataset_augmented = torchvision.datasets.ImageFolder(
        root=data_path_train,
        transform=transformation_train_augmentation
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path_test,
        transform=transformation_test
    )

    # 30% images taken out of the train set to create the validation set
    train_size = int(0.7 * len(train_dataset_normal))
    val_size = len(train_dataset_normal) - train_size

    Train_set_after_validation_excluded, val_dataset = torch.utils.data.random_split(train_dataset_normal,
                                                                                     [train_size, val_size])

    Train_set_augmented = torch.utils.data.ConcatDataset([Train_set_after_validation_excluded, train_dataset_augmented])

    # Set Batch size to whatever suits the task
    Train_loader = torch.utils.data.DataLoader(
        Train_set_augmented,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    Validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    Test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=460,
        num_workers=0,
        shuffle=False
    )

    return Train_loader, Validation_loader, Test_loader


train_loader, val_loader, test_loader = load_dataset(Train_folder, Test_folder)

batch_size = train_loader.batch_size


class Net(nn.Module):
    """
    in_channels=1 since images are grayscale. Change to "3" if the task requires so (RGB images)
    Both nn.BatchNorm2d and nn.Dropout(p=0.3) prevent overfitting quite well
    """
    def __init__(self, num_classes=len(classes)):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module('Conv1', nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, padding=2))
        self.layer1.add_module('BN1', nn.BatchNorm2d(num_features=8, eps=1e-05, momentum=0.1, affine=True,
                                                     track_running_stats=True))

        self.layer1.add_module('Pool1', nn.MaxPool2d(kernel_size=4))
        self.layer1.add_module('ReLu1', nn.LeakyReLU(inplace=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('Conv2', nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, padding=2))
        self.layer2.add_module('BN2', nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True,
                                                     track_running_stats=True))

        self.layer2.add_module('Pool2', nn.MaxPool2d(kernel_size=4))
        self.layer2.add_module('ReLu2', nn.LeakyReLU(inplace=False))

        self.fully_connected1 = nn.Linear(8 * 8 * 16, 256)

        self.fully_connected2 = nn.Linear(256, num_classes)

        self.pool = nn.MaxPool2d(kernel_size=4)
        self.dropout = nn.Dropout(p=0.3)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)

        x = x.view(-1, 8 * 8 * 16)

        x = self.ReLU(self.dropout(self.fully_connected1(x)))

        x = self.fully_connected2(x)

        return x


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

model = Net(num_classes=len(classes)).to(device)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    correct_train = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = loss_criteria(output, target)

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        correct_train += torch.sum(target == predicted).item()

        print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))

    avg_loss_train = train_loss / (batch_idx + 1)

    print('Training set: loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss_train, correct_train, len(train_loader.dataset),
        100. * correct_train / len(train_loader.dataset)))
    return avg_loss_train


def Validation(Selected_model, Device_used, val_load):
    Selected_model.eval()
    val_loss = 0
    correct_val = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in val_load:
            batch_count += 1
            data, target = data.to(Device_used), target.to(Device_used)

            output = Selected_model(data)

            val_loss += loss_criteria(output, target).item()

            _, predicted = torch.max(output.data, 1)
            correct_val += torch.sum(target == predicted).item()

    avg_loss_val = val_loss / batch_count
    print('Validation set: loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss_val, correct_val, len(val_load.dataset),
        100. * correct_val / len(val_load.dataset)))

    return avg_loss_val


def Check_accuracy(model, device, test_loader):
    model.eval()
    correct_test = 0
    test_loss = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += loss_criteria(output, target).item()

            _, predicted = torch.max(output.data, 1)
            correct_test += torch.sum(target == predicted).item()

    avg_loss_test = test_loss / batch_count

    print('Test set: loss: {:.6f} AccuracyTest: {}/{} ({:.0f}%)\n'.format(
        avg_loss_test, correct_test, len(test_loader.dataset),
        100. * correct_test / len(test_loader.dataset)))


# Optimizer selection, as well as hyperparameter tuning
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_criteria = nn.CrossEntropyLoss()

# set the number of epochs. 96% accuracy acquired at 50th epoch and 46 batch size.



for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    Validation(model, device, val_loader)
Check_accuracy(model, device, test_loader)
