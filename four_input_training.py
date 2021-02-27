import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torchvision.transforms import transforms

from models.net_1357 import FourInputNet
from dicom_dataset import DicomDataset1357
from ssim_loss import SSIM

import time
import numpy as np

image_size = 64
batch_size = 10
learning_rate = 0.001
num_of_epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(image_size),
    transforms.Normalize((0.5,),(0.5,))
])

dataset = DicomDataset1357("../slike/", transform=transformations)

train_size = int(0.8 * len(dataset))
validation_size = int(0.15 * len(dataset))
test_size = len(dataset)-train_size-validation_size

train_set, validation_set, test_set = random_split(dataset, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set,batch_size=batch_size, shuffle=True)


model = FourInputNet()
model.to(device)

criterion_ssim = SSIM()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.876543, patience=5, threshold_mode="abs" ,threshold=0.0001 ,verbose=True)

for epoch in range(num_of_epochs):
    start_time = time.time()
    #Training
    model.train()
    training_losses = []
    for iteration, (one_img,three_image,five_image,seven_image, expcted_img) in enumerate(train_loader):
        one_img = one_img.to(device=device)
        three_image = three_image.to(device=device)
        five_image = five_image.to(device=device)
        seven_image = seven_image.to(device=device)
        
        expcted_img = expcted_img.to(device=device)
        
        optimizer.zero_grad()

        output = model(one_img,three_image,five_image,seven_image)

        loss = 1 - criterion_ssim(output, expcted_img)

        loss_value = loss.item()
        
        loss.backward()
        
        optimizer.step()
        
        training_losses.append(loss_value)
    avg_training_losses = np.array(training_losses).mean()
    print("Epoch {} Training Completed: Train Avg. SSIM Loss: {:.4f}".format(epoch+1, avg_training_losses))
    
    #Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for one_img,three_image,five_image,seven_image, expcted_img in validation_loader:
            one_img = one_img.to(device=device)
            three_image = three_image.to(device=device)
            five_image = five_image.to(device=device)
            seven_image = seven_image.to(device=device)

            expcted_img = expcted_img.to(device=device)
            
            output = model(one_img,three_image,five_image,seven_image)

            loss = 1 - criterion_ssim(output, expcted_img)

            val_losses.append(loss.item())
    avg_val_losses = np.array(val_losses).mean()
    print("Epoch {} Validation Completed: Validation Avg. Loss: {:.4f}".format(epoch+1, avg_val_losses))
    end_time = time.time() - start_time
    print("This epoch took {:.2f} seconds to complete".format(end_time))
    #scheduler.step(avg_val_losses)

#Saving trained model
torch.save(model.state_dict(),"1357model.pt")
print("Saving model")
