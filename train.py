import copy
import time
import torch
import torch.nn as nn
import math
import stat_model 
import baseline_model 

def train_model(model, dataset, config):
    # Extract configuration parameters
    loss_fn = config["loss_fn"]
    optimizer = config["optimizer"]
    lr = config["learning_rate"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    scheduler = config.get("scheduler", None)

    # Create data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set up the optimizer and scheduler
    optimizer = optimizer(model.parameters(), lr=lr)
    if scheduler is not None:
        scheduler = scheduler(optimizer)

    # Train the model
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i, (inputs, attention_masks, labels, stat_vec) in enumerate(data_loader):
            # Zero out gradients
            optimizer.zero_grad()

            # Convert inputs to tensors
            inputs = torch.tensor(inputs)
            attention_masks = torch.tensor(attention_masks)
            labels = torch.tensor(labels)
            stat_vec = torch.tensor(stat_vec)

            # Compute outputs and loss
            outputs = model(inputs, attention_masks, stat_vec)
            loss = loss_fn(outputs, labels)

            # Backpropagate and update weights
            loss.backward()
            optimizer.step()

            # Update total loss
            total_loss += loss.item()

        # Output average loss for epoch
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg. Loss: {avg_loss:.4f}")

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

    print("Training complete!")
