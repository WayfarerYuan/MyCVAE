import torch

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = torch.device('mps')

class CVAE(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim, latent_dim):
        super(CVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + label_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # 输出均值和方差
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + label_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出的像素值在[0, 1]之间
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, label):
        combined_input = torch.cat([x, label], dim=1)
        mu, logvar = self.encoder(combined_input).chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(torch.cat([z, label], dim=1)), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets

def train(model, dataloader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(dataloader):
        data = data.view(-1, 784)  # Flatten the images
        labels = one_hot(labels, 10)  # One hot encode the labels
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data, labels)
        loss = loss_function(recon_batch, data, mu, logvar)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(dataloader.dataset)))


input_dim = 784  # 28*28, size of MNIST images
label_dim = 10  # Number of classes in MNIST
hidden_dim = 400  
latent_dim = 10  

model = CVAE(input_dim, label_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Download and load the data
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Training the model
epochs = 10
for epoch in range(1, epochs + 1):
    train(model, train_loader, optimizer, epoch)

def generate_digit(model, z, label):
    # model: trained model
    # z: latent variable, torch.Tensor of shape (latent_dim,)
    # label: class label, integer

    # Perform one-hot encoding on the label
    label_onehot = torch.zeros(10).to(z.device)  # Assuming there are 10 classes
    label_onehot[label] = 1.

    # Concatenate z and label
    z_and_label = torch.cat([z, label_onehot])

    # Pass through the decoder
    digit = model.decoder(z_and_label)

    # Reshape and visualize the digit
    digit_reshaped = digit.view(28, 28).detach().cpu()
    plt.imshow(digit_reshaped, cmap='gray')
    # plt.show()
    return digit

# Generate 0-9 in one row
fig, axes = plt.subplots(1, 10, figsize=(20, 2))  # Creates a grid of 10 subplots
for i in range(10):
    z = torch.randn(latent_dim).to(device)  # Randomly sample z from normal distribution
    label = i  # Class label
    digit = generate_digit(model, z, label)  # Generate a digit

    # Reshape and visualize the digit
    digit_reshaped = digit.view(28, 28).detach().cpu().numpy()
    axes[i].imshow(digit_reshaped, cmap='gray')  # Display the digit in the i-th subplot
    axes[i].set_title(f'Generated Digit: {label}')  # Set the title of the subplot to the label
    axes[i].axis('off')  # Remove axis

plt.show()  # Display the full plot








