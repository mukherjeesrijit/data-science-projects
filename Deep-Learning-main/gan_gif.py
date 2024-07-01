import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

# Define the generator model with initial weights
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Linear(1, 1)
        # Initialize weights with a specific value
        nn.init.normal_(self.linear.weight, mean=1.0, std=0.0)
        # Initialize bias with a specific value
        nn.init.constant_(self.linear.bias, 1.0)

    def forward(self, x):
        return self.linear(x)

# Define the discriminator model with initial weights
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(1, 1)
        # Initialize weights with a specific value
        nn.init.normal_(self.linear.weight, mean=-0.5, std=0.1)
        # Initialize bias with a specific value
        nn.init.constant_(self.linear.bias, 3.0)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Define loss function and optimizers
criterion = nn.BCELoss()
optimizer_discriminator = optim.SGD(discriminator.parameters(), lr=0.01)
optimizer_generator = optim.SGD(generator.parameters(), lr=0.01)

# Training parameters
steps = 1000
size = 100
real_data_mean = 3.0
real_data_std = 1.0
update_interval = 50  # Update the plot every 50 steps
gif_frames = 50

# Create subplots for dynamic plotting
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # Adjust bottom to make space for button

# Function to update the plot for one step
def update_plot(frame):
    step = frame * update_interval
    for _ in range(update_interval):
        # Step 1: Generate fake data
        noise = torch.randn(size, 1)
        fake_data = generator(noise)

        # Step 2: Generate real data
        # Generate real data using numpy
        real_data_np = np.random.normal(loc=real_data_mean, scale=real_data_std, size=(size, 1))
        real_data = torch.tensor(real_data_np, dtype=torch.float32)

        # Repeat the process to generate samples for the real data
        real_data = real_data.repeat(1, 1)

        # Step 3: Train discriminator
        optimizer_discriminator.zero_grad()
        output_real = discriminator(real_data)
        loss_real = criterion(output_real, torch.ones_like(output_real))
        output_fake = discriminator(fake_data.detach())
        loss_fake = criterion(output_fake, torch.zeros_like(output_fake))
        loss_discriminator = loss_real + loss_fake
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Step 4: Train generator
        optimizer_generator.zero_grad()
        output_fake = discriminator(fake_data)
        loss_generator = criterion(output_fake, torch.ones_like(output_fake))
        loss_generator.backward()
        optimizer_generator.step()

    # Convert samples to numpy arrays
    fake_samples = fake_data.detach().numpy()
    real_samples = real_data.numpy()

    # Calculate decision boundary
    samples_range = np.linspace(min(min(fake_samples), min(real_samples)), max(max(fake_samples), max(real_samples)), 100)
    boundary = discriminator(torch.tensor(samples_range[:, np.newaxis])).detach().numpy().flatten()

    # Clear previous plot
    ax.clear()

    # Plot the histograms
    ax.hist(fake_samples, bins=50, density=True, alpha=0.5, label='Fake Samples')
    ax.hist(real_samples, bins=50, density=True, alpha=0.5, label='Real Samples')

    # Plot the normal density for fake data using updated generator parameters
    fake_density = (1 / (generator.linear.weight.item() * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((samples_range - generator.linear.bias.item()) / generator.linear.weight.item()) ** 2)
    ax.plot(samples_range, fake_density, label='Fake Data Density', color='blue')

    # Plot the normal density for real data
    real_density = (1 / (real_data_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((samples_range - real_data_mean) / real_data_std) ** 2)
    ax.plot(samples_range, real_density, label='Real Data Density', color='green')

    # Plot the decision boundary
    ax.plot(samples_range, boundary, label='Decision Boundary', color='red')

    # Set x and y limits based on the data
    ax.set_xlim(min(samples_range), max(samples_range))
    ax.set_ylim(0, 1 + max(max(fake_density), max(real_density), max(boundary)))

    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'GAN Training for Step {step}')
    ax.legend()

# Create animation
ani = FuncAnimation(fig, update_plot, frames=gif_frames, interval=500)  # Interval in milliseconds

# Save animation as GIF
ani.save('gan_training.gif', writer='pillow')

# Show the plot
plt.show()
