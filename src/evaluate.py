import os
import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from model import create_model


def evaluate_model(model_path, test_data_dir, img_size=(512, 512), batch_size=32):
    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Load test dataset
    test_dataset = datasets.ImageFolder(
        root=test_data_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Load the trained model
    model = create_model(num_classes=3)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Evaluate the model
    true_classes = []
    predicted_classes = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            true_classes.extend(labels.cpu().numpy())
            predicted_classes.extend(preds.cpu().numpy())

    # Generate classification report
    class_labels = list(test_dataset.class_to_idx.keys())
    report = classification_report(
        true_classes, predicted_classes, target_names=class_labels)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == "__main__":
    model_path = 'models/guava_disease_model.pth'  # Update with your model path
    test_data_dir = 'data/processed/test'  # Update with your test data directory
    evaluate_model(model_path, test_data_dir)
