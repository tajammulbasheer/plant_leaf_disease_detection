import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@torch.no_grad()
def evaluate_classifier(model, dataloader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
    }
