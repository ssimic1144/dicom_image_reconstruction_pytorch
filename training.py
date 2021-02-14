import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torchvision.transforms import transforms

from nn_model import Net
from dicom_dataset import DicomDataset
from ssim_loss import SSIM

import time

image_size = 64
batch_size = 60
learning_rate = 0.001
num_of_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(image_size),
    transforms.Normalize((0.5), (0.5))
])

dataset = DicomDataset("../slike/", transform=transformations)

train_size = int(0.8 * len(dataset))
validation_size = int(0.15 * len(dataset))
test_size = len(dataset)-train_size-validation_size

train_set, validation_set, test_set = random_split(dataset, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set,batch_size=batch_size, shuffle=True)


model = Net()
model.to(device)

criterion_ssim = SSIM()
criterion_mse = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(num_of_epochs):
    start_time = time.time()
    #Training
    model.train()
    epoch_loss = 0
    n_samples = 0
    for iteration, (prev_img, next_img, expcted_img) in enumerate(train_loader):
        prev_img = prev_img.to(device=device)
        next_img = next_img.to(device=device)
        expcted_img = expcted_img.to(device=device)

        optimizer.zero_grad()

        output = model(prev_img,next_img)

        loss = 1 - criterion_ssim(output, expcted_img)
        #loss = criterion_mse(output, expcted_img)

        loss_value = loss.item()
        
        loss.backward()
        
        optimizer.step()

        epoch_loss += loss_value * prev_img.shape[0]
        n_samples += prev_img.shape[0]
        #print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch+1, iteration, len(train_loader), loss_value))
    print("Epoch {} Complete: Train Avg. Loss: {:.4f}".format(epoch+1, epoch_loss/n_samples))
    
    #Validation
    model.eval()
    total_loss = 0
    n_samples = 0
    with torch.no_grad():
        for prev_img, next_img, expcted_img in validation_loader:
            prev_img, next_img = prev_img.to(device),next_img.to(device)
            expcted_img = expcted_img.to(device)

            output = model(prev_img, next_img)

            loss = 1 - criterion_ssim(output, expcted_img)

            total_loss += loss.item() * prev_img.shape[0]
            n_samples += prev_img.shape[0]
    print("Epoch {} Validation Complete: Validation Avg. Loss: {:.4f}".format(epoch+1, total_loss/n_samples))
    end_time = time.time() - start_time
    print("This epoch took {:.2f} seconds to complete".format(end_time))
