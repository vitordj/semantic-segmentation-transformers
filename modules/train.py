import torch
import pandas as pd
import numpy as np
from datetime import datetime
import torch.nn as nn
from modules.metrics import adjust_labels_for_metrics, calculate_metrics_for_classes

def train_and_validate_segformer(train_dataloader, valid_dataloader, model
                                 , optimizer, criterion, device, epochs=15
                                 , scheduler=None, early_stopping_patience=10, save_model=False):
    """
    Function to train and validate a Segformer model for semantic segmentation.
    :param train_dataloader: DataLoader for training dataset
    :param valid_dataloader: DataLoader for validation dataset
    :param model: Segformer model
    :param optimizer: Optimizer
    :param criterion: Loss function
    :param device: Device to use for training
    :param epochs: Number of epochs to train
    :param scheduler: Scheduler for learning rate
    :param early_stopping_patience: Patience for early stopping
    :param save_model: Whether to save the best model
    :return: A tuple with the resulting DataFrame and a OrderedDict with the best model state_dict (if save_model=1, else None)
    """
    best_loss = float('inf')
    early_stopping_counter = 0
    metrics_per_epoch = []

    for epoch in range(epochs):
        model.train()
        train_loss_accum = 0
        epoch_metrics = {'epoch': epoch}

        for batch in train_dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            labels = torch.where(labels == 255, torch.tensor(1, device=labels.device), labels)

            # Se criterion é None, usa a perda padrão
            if criterion:
                loss = criterion(upsampled_logits, labels)
            else:
                loss = outputs.loss

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

            if scheduler.__class__.__name__ == 'OneCycleLR':
                scheduler.step()

        if scheduler.__class__.__name__ == 'PolynomialLR':
            scheduler.step()

        avg_train_loss = train_loss_accum / len(train_dataloader)

        model.eval()
        valid_loss_accum = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in valid_dataloader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                logits = outputs.logits
                upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                labels = torch.where(labels == 255, torch.tensor(1, device=labels.device), labels)

                # Se criterion é None, usa a perda padrão
                if criterion:
                    loss = criterion(upsampled_logits, labels)
                else:
                    loss = outputs.loss

                valid_loss_accum += loss.item()

                predicted = upsampled_logits.argmax(dim=1)
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Agregar predições para cálculo das métricas
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        class_list = [1, 0]  # Atualize conforme necessário

        adjusted_predicted, adjusted_labels = adjust_labels_for_metrics(all_preds, all_labels)

        metrics = calculate_metrics_for_classes(adjusted_predicted, adjusted_labels, class_list)

        for class_id, class_metrics in metrics.items(): # processa para cada ID
            print(f"Metrics for Class {class_id}: Precision: {class_metrics['precision']}, Recall: {class_metrics['recall']}, F1-Score: {class_metrics['f1-score']}")

            epoch_metrics.update({
                f'precision_{class_id}': class_metrics['precision'],
                f'recall_{class_id}': class_metrics['recall'],
                f'f1_score_{class_id}': class_metrics['f1-score']
            })

        avg_valid_loss = valid_loss_accum / len(valid_dataloader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}")

        # Update epoch losses
        epoch_metrics.update({'training_loss': avg_train_loss
                              , 'validation_loss': avg_valid_loss})
        metrics_per_epoch.append(epoch_metrics)

        # Early Stopping Check
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            early_stopping_counter = 0
            if save_model:
                best_model = model.state_dict()
                print("Modelo salvo como melhor modelo.")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    if save_model:
        return pd.DataFrame(metrics_per_epoch), best_model
    else:
        return pd.DataFrame(metrics_per_epoch), None

def train_and_validate_maskformer(train_dataloader, valid_dataloader, model
                                 , optimizer, preprocessor, device, epochs=15
                                 , scheduler=None, early_stopping_patience=10, save_model=False):
    """
    Function to train and validate a Segformer model for semantic segmentation.
    :param train_dataloader: DataLoader for training dataset
    :param valid_dataloader: DataLoader for validation dataset
    :param model: Maskformer/Mask2Former model
    :param optimizer: Optimizer
    :param criterion: Loss function
    :param device: Device to use for training
    :param epochs: Number of epochs to train
    :param scheduler: Scheduler for learning rate
    :param early_stopping_patience: Patience for early stopping
    :param save_model: Whether to save the best model
    :return: A tuple with the resulting DataFrame and a OrderedDict with the best model state_dict (if save_model=1, else None)

    """
    best_loss = float('inf')
    early_stopping_counter = 0
    metrics_per_epoch = []

    for epoch in range(epochs):
        model.train()
        train_loss_accum = 0
        epoch_metrics = {'epoch': epoch}

        for batch in train_dataloader:
            optimizer.zero_grad()
            if 'oneformer' in model.name_or_path:
                for key in batch:
                    batch[key] = batch[key].to(device)
                outputs = model(**batch)
            else:
                pixel_values = batch["pixel_values"].to(device)
                outputs = model(
                    pixel_values=pixel_values,
                    mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                    class_labels=[labels.to(device) for labels in batch["class_labels"]],
                )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

            if scheduler.__class__.__name__ == 'OneCycleLR':
                scheduler.step()

        if scheduler.__class__.__name__ == 'PolynomialLR':
            scheduler.step()

        avg_train_loss = train_loss_accum / len(train_dataloader)

        model.eval()
        valid_loss_accum = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in valid_dataloader:
                if 'oneformer' in model.name_or_path:
                    for key in batch:
                        batch[key] = batch[key].to(device)
                    outputs = model(**batch)
                else:
                    pixel_values = batch["pixel_values"].to(device)
                    outputs = model(
                        pixel_values=pixel_values,
                        mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                        class_labels=[labels.to(device) for labels in batch["class_labels"]],
                    )
                valid_loss_accum += outputs.loss.item()
                original_images = batch["pixel_values"]
                target_sizes = [(image.shape[1], image.shape[2]) for image in original_images]
                predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                            target_sizes=target_sizes)
                _predicted_segmentation_maps = np.array([i.cpu().numpy() for i in predicted_segmentation_maps])
                ground_truth_segmentation_maps = batch["mask_labels"].cpu().numpy() if isinstance(batch["mask_labels"], torch.Tensor) else batch["mask_labels"]
                _ground_truth_segmentation_maps = np.array([i[1] for i in ground_truth_segmentation_maps])
                all_preds.append(_predicted_segmentation_maps)
                all_labels.append(_ground_truth_segmentation_maps)

        # Agregar predições para cálculo das métricas
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        class_list = [1, 0]  # Atualize conforme necessário

        metrics = calculate_metrics_for_classes(all_preds, all_labels, class_list)
        for class_id, class_metrics in metrics.items(): # processa para cada ID
            print(f"Metrics for Class {class_id}: Precision: {class_metrics['precision']}, Recall: {class_metrics['recall']}, F1-Score: {class_metrics['f1-score']}")

            epoch_metrics.update({
                f'precision_{class_id}': class_metrics['precision'],
                f'recall_{class_id}': class_metrics['recall'],
                f'f1_score_{class_id}': class_metrics['f1-score']
            })

        avg_valid_loss = valid_loss_accum / len(valid_dataloader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}")

        # Update epoch losses
        epoch_metrics.update({'training_loss': avg_train_loss
                              , 'validation_loss': avg_valid_loss})
        metrics_per_epoch.append(epoch_metrics)

        # Early Stopping Check
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            early_stopping_counter = 0
            if save_model:
                best_model = model.state_dict()
                print("Modelo salvo como melhor modelo.")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    if save_model:
        return pd.DataFrame(metrics_per_epoch), best_model
    else:
        return pd.DataFrame(metrics_per_epoch), None