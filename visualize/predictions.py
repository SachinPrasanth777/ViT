import matplotlib.pyplot as plt
import torch

def show_predictions(model, data_loader, device, num_images=20):
    model.eval()
    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)
    plt.figure(figsize=(12, 4))
    for i in range(num_images):
        plt.subplot(4, 5, i+1)
        plt.imshow(images[i].squeeze().cpu().numpy(), cmap='gray')
        plt.title(f"Pred: {preds[i].item()}\nTrue: {labels[i].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
