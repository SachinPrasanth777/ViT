import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from config import settings
from models.vit import VisionTransformer
from train.train import train_one_epoch, validate
from visualize.predictions import show_predictions

transform = transforms.Compose([
    transforms.RandomVerticalFlip(0.1),
    transforms.RandomHorizontalFlip(0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=settings.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=settings.batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VisionTransformer(
    image_size=settings.image_size,
    patch_size=settings.patch_size,
    in_channels=settings.channels,
    embed_dim=settings.embed_dim,
    num_heads=settings.num_heads,
    mlp_dim=settings.mlp_dim,
    num_classes=settings.num_classes,
    transformer_units=settings.transformer_units
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs, eta_min=0.0001)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_acc = 0
patience, counter = 5, 0

for epoch in range(settings.epochs):
    print(f"\nEpoch {epoch+1}")
    loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_acc = validate(model, val_loader, device)
    scheduler.step()
    print(f"Train Loss: {loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_vit_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

print(f"\nBest validation accuracy: {best_acc:.2f}%")
show_predictions(model, val_loader, device)
