import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define a complex CNN model
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        
        # Initial convolution
        self.conv_init = nn.Conv2d(3, 64, 3, padding=1)
        self.bn_init = nn.BatchNorm2d(64)
        
        # Block 1 - 64 channels
        self.conv1_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        # Block 2 - 128 channels
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2),
            nn.BatchNorm2d(128)
        )
        
        # Block 3 - 256 channels
        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.downsample3 = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2),
            nn.BatchNorm2d(256)
        )
        
        # Block 4 - 512 channels
        self.conv4_1 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.downsample4 = nn.Sequential(
            nn.Conv2d(256, 512, 1, stride=2),
            nn.BatchNorm2d(512)
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        
        # Activation and regularization
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.4)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.relu(self.bn_init(self.conv_init(x)))
        
        # Block 1 with residual
        identity = x
        out = self.relu(self.bn1_1(self.conv1_1(x)))
        out = self.bn1_2(self.conv1_2(out))
        out += identity
        out = self.relu(out)
        
        # Block 2 with residual
        identity = self.downsample2(out)
        out = self.relu(self.bn2_1(self.conv2_1(out)))
        out = self.bn2_2(self.conv2_2(out))
        out += identity
        out = self.relu(out)
        
        # Block 3 with residual
        identity = self.downsample3(out)
        out = self.relu(self.bn3_1(self.conv3_1(out)))
        out = self.bn3_2(self.conv3_2(out))
        out += identity
        out = self.relu(out)
        
        # Block 4 with residual
        identity = self.downsample4(out)
        out = self.relu(self.bn4_1(self.conv4_1(out)))
        out = self.bn4_2(self.conv4_2(out))
        out += identity
        out = self.relu(out)
        
        # Global average pooling
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        
        # Fully connected layers
        out = self.relu(self.bn_fc1(self.fc1(out)))
        out = self.dropout(out)
        out = self.relu(self.bn_fc2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

# Load CIFAR-10 dataset with strong data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load full training set
full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)

# Split training set into train and validation
train_size = int(0.9 * len(full_trainset))
val_size = len(full_trainset) - train_size
trainset, valset = random_split(full_trainset, [train_size, val_size])

# Create validation set with test transform (no augmentation)
valset.dataset.transform = transform_test

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)

testloader = DataLoader(testset, batch_size=100, shuffle=False)
valloader = DataLoader(valset, batch_size=100, shuffle=False)

# Function to calculate difficulty score for each sample
def calculate_difficulty_scores(model, dataset, device):
    """Calculate difficulty based on prediction entropy and loss"""
    model.eval()
    difficulties = []
    
    # Use no augmentation for difficulty estimation
    temp_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Access the underlying dataset if it's a Subset
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = range(len(dataset))
    
    with torch.no_grad():
        for idx, i in enumerate(indices):
            if idx % 5000 == 0:
                print(f"Processing sample {idx}/{len(indices)}")
            
            img, label = base_dataset.data[i], base_dataset.targets[i]
            img = transforms.ToPILImage()(img)
            img = temp_transform(img).unsqueeze(0).to(device)
            label = torch.tensor([label]).to(device)
            
            output = model(img)
            
            # Calculate loss
            loss = nn.CrossEntropyLoss()(output, label).item()
            
            # Calculate entropy of predictions
            probs = torch.softmax(output, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
            
            # Combine loss and entropy for difficulty score
            difficulty = 0.7 * loss + 0.3 * entropy
            difficulties.append(difficulty)
    
    return np.array(difficulties)

# Training function with validation
def train_model(model, trainloader, valloader, testloader, epochs, device, method_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    train_losses = []
    val_accuracies = []
    test_accuracies = []
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(trainloader)
        train_losses.append(avg_train_loss)
        train_acc = 100 * correct / total
        
        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Evaluate on test set
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        test_accuracies.append(test_acc)
        
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f'{method_name} - Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return train_losses, val_accuracies, test_accuracies

# Main execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Model parameters: {sum(p.numel() for p in ComplexCNN().parameters()):,}")

# Parameters
epochs = 40
batch_size = 128

# Method 1: Standard Random Order Training
print("\n=== Training with Standard Random Order ===")
model_random = ComplexCNN().to(device)
trainloader_random = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
random_losses, random_val_accs, random_test_accs = train_model(model_random, trainloader_random, valloader, testloader, epochs, device, "Random")

# Method 2: Curriculum Learning
print("\n=== Training with Curriculum Learning ===")
# First, train a preliminary model to estimate difficulty
print("Training preliminary model for difficulty estimation...")
preliminary_model = ComplexCNN().to(device)
preliminary_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
preliminary_optimizer = optim.SGD(preliminary_model.parameters(), lr=0.1, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Train for 8 epochs for better difficulty estimation
for epoch in range(8):
    preliminary_model.train()
    for inputs, labels in preliminary_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        preliminary_optimizer.zero_grad()
        outputs = preliminary_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        preliminary_optimizer.step()
    print(f"Preliminary training epoch {epoch+1}/8 complete")

# Calculate difficulty scores
print("Calculating difficulty scores for all samples...")
difficulty_scores = calculate_difficulty_scores(preliminary_model, trainset, device)

# Sort samples by difficulty (easy to hard)
sorted_indices = np.argsort(difficulty_scores)

# Create curriculum learning with more sophisticated schedule
model_curriculum = ComplexCNN().to(device)
optimizer_curr = optim.SGD(model_curriculum.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler_curr = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_curr, T_0=10, T_mult=2)
curriculum_losses = []
curriculum_val_accs = []
curriculum_test_accs = []

for epoch in range(epochs):
    # More gradual curriculum progression
    if epoch < 5:
        proportion = 0.3  # 30% easiest
    elif epoch < 10:
        proportion = 0.5  # 50% easiest
    elif epoch < 15:
        proportion = 0.7  # 70% easiest
    elif epoch < 25:
        proportion = 0.85  # 85% easiest
    else:
        proportion = 1.0  # All data
    
    num_samples = int(proportion * len(trainset))
    current_indices = sorted_indices[:num_samples].tolist()
    
    # Add some random hard samples to prevent overfitting to easy samples
    if proportion < 1.0:
        num_hard = int(0.1 * num_samples)  # Add 10% hard samples
        hard_indices = sorted_indices[-num_hard:].tolist()
        current_indices = list(set(current_indices + hard_indices))
    
    current_subset = Subset(trainset, current_indices)
    curriculum_loader = DataLoader(current_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Train for one epoch
    model_curriculum.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in curriculum_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer_curr.zero_grad()
        outputs = model_curriculum(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_curr.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_train_loss = running_loss / len(curriculum_loader)
    curriculum_losses.append(avg_train_loss)
    train_acc = 100 * correct / total
    
    # Evaluate on validation set
    model_curriculum.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_curriculum(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = 100 * correct / total
    curriculum_val_accs.append(val_acc)
    
    # Evaluate on test set
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_curriculum(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    curriculum_test_accs.append(test_acc)
    
    scheduler_curr.step()
    
    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f'Curriculum - Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%, Using {proportion*100:.0f}% of data')

# Plotting results
plt.figure(figsize=(15, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), random_losses, 'b-', label='Random Order', linewidth=2)
plt.plot(range(1, epochs+1), curriculum_losses, 'r--', label='Curriculum Learning', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Training Loss Comparison - Complex CNN', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Plot validation accuracy with smoothing
from scipy.ndimage import gaussian_filter1d
smooth_random = gaussian_filter1d(random_val_accs, sigma=1)
smooth_curriculum = gaussian_filter1d(curriculum_val_accs, sigma=1)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), smooth_random, 'b-', label='Random Order', linewidth=2)
plt.plot(range(1, epochs+1), smooth_curriculum, 'r--', label='Curriculum Learning', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Accuracy (%)', fontsize=12)
plt.title('Validation Accuracy Comparison - Complex CNN', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Add curriculum stage indicators
for x, label in [(5, '30%'), (10, '50%'), (15, '70%'), (25, '85%')]:
    plt.axvline(x=x, color='gray', linestyle=':', alpha=0.5)
    plt.text(x, plt.ylim()[0]+1, label, ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('complex_cnn_curriculum_vs_random.png', dpi=150)
plt.show()

# Comprehensive results analysis
print("\n=== Final Results ===")
print(f"Random Order - Final Validation Accuracy: {random_val_accs[-1]:.2f}%")
print(f"Curriculum Learning - Final Validation Accuracy: {curriculum_val_accs[-1]:.2f}%")
print(f"Difference: {random_val_accs[-1] - curriculum_val_accs[-1]:.2f}% (positive means random is better)")

# Statistical analysis over different windows
for window in [5, 10]:
    avg_random = np.mean(random_val_accs[-window:])
    avg_curriculum = np.mean(curriculum_val_accs[-window:])
    print(f"\nAverage validation accuracy (last {window} epochs):")
    print(f"Random Order: {avg_random:.2f}%")
    print(f"Curriculum Learning: {avg_curriculum:.2f}%")
    print(f"Difference: {avg_random - avg_curriculum:.2f}%")

# Peak and convergence analysis
peak_random = max(random_val_accs)
peak_curriculum = max(curriculum_val_accs)
print(f"\nPeak validation accuracy achieved:")
print(f"Random Order: {peak_random:.2f}% (epoch {random_val_accs.index(peak_random)+1})")
print(f"Curriculum Learning: {peak_curriculum:.2f}% (epoch {curriculum_val_accs.index(peak_curriculum)+1})")
print(f"Difference: {peak_random - peak_curriculum:.2f}%")

# Convergence speed
threshold = 85  # When model reaches 85% validation accuracy
random_convergence = next((i for i, acc in enumerate(random_val_accs) if acc >= threshold), -1) + 1
curr_convergence = next((i for i, acc in enumerate(curriculum_val_accs) if acc >= threshold), -1) + 1
print(f"\nEpochs to reach {threshold}% validation accuracy:")
print(f"Random Order: {random_convergence if random_convergence > 0 else 'Not reached'}")
print(f"Curriculum Learning: {curr_convergence if curr_convergence > 0 else 'Not reached'}")

# Final test accuracy comparison
print(f"\n=== Test Set Performance ===")
print(f"Random Order - Final Test Accuracy: {random_test_accs[-1]:.2f}%")
print(f"Curriculum Learning - Final Test Accuracy: {curriculum_test_accs[-1]:.2f}%")
print(f"Difference: {random_test_accs[-1] - curriculum_test_accs[-1]:.2f}%")