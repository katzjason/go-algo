from engine import Board, board_to_tensor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GoChunkedDataset
from model import ResNet
import wandb
import os


def main():

    # Configuration
    CHUNK_DIR = "./train_data"  # adjust to match your EC2 directory
    BATCH_SIZE = 64
    EPOCHS = 40
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_WANDB = True
    PATIENCE = 5

    # Initializing WandB
    if USE_WANDB:
        wandb.init(project="go-ai", config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "model": "ResNet",
            "input_shape": "[10, 9, 9]",
        })

    # Load Dataset
    dataset = GoChunkedDataset(CHUNK_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = ResNet().to(DEVICE)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_policy_loss = float('inf')
    consec_declines = 0

    # Training Loop
    os.makedirs("checkpoints", exist_ok=True)
    for epoch in range(EPOCHS):
        model.train()
        total_policy_loss = 0
        total_value_loss = 0
        correct_policy = 0
        total_samples = 0

        for X, y_policy, y_value in dataloader:
            X = X.to(DEVICE)
            y_policy = y_policy.to(DEVICE)
            y_value = y_value.to(DEVICE).float()

            optimizer.zero_grad()

            policy_logits, value_pred = model(X)  # shape: (batch, 81), (batch, 1)

            loss_policy = policy_loss_fn(policy_logits, y_policy)
            loss_value = value_loss_fn(value_pred.squeeze(), y_value)

            loss = loss_policy + loss_value
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()

            preds = torch.argmax(policy_logits, dim=1)
            correct_policy += (preds == y_policy).sum().item()
            total_samples += y_policy.size(0)

        avg_policy_loss = total_policy_loss / len(dataloader)
        avg_value_loss = total_value_loss / len(dataloader)
        policy_accuracy = correct_policy / total_samples * 100

        print(f"Epoch {epoch + 1}/{EPOCHS} | Policy Acc: {policy_accuracy:.2f}% | "
            f"Policy Loss: {avg_policy_loss:.4f} | Value Loss: {avg_value_loss:.4f}")

        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f"checkpoints/go_resnet_epoch{epoch+1}.pt")

        if USE_WANDB:
            wandb.log({
                "epoch": epoch + 1,
                "policy_accuracy": policy_accuracy,
                "policy_loss": avg_policy_loss,
                "value_loss": avg_value_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "grad_norm": grad_norm,
            })
        
        if avg_policy_loss <= best_policy_loss:
            best_policy_loss = avg_policy_loss
            consec_declines = 0
        else:
            consec_declines += 1

        if consec_declines >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
            break


    # Evaluation on val set
    val_dataset = GoChunkedDataset("./test_data")
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model.eval()
    correct_policy = 0
    total_value_loss = 0
    total_samples = 0

    with torch.no_grad():
        for X, y_policy, y_value in val_loader:
            X = X.to(DEVICE)
            y_policy = y_policy.to(DEVICE)
            y_value = y_value.to(DEVICE).float()

            policy_logits, value_pred = model(X)

            preds = torch.argmax(policy_logits, dim=1)
            correct_policy += (preds == y_policy).sum().item()
            total_value_loss += nn.MSELoss()(value_pred.squeeze(), y_value).item()
            total_samples += y_policy.size(0)

    policy_acc = correct_policy / total_samples * 100
    avg_val_loss = total_value_loss / len(val_loader)

    if USE_WANDB:
        wandb.log({
            "val_policy_accuracy": policy_acc,
            "val_value_loss": avg_val_loss
    })

    print(f"Validation — Policy Acc: {policy_acc:.2f}% | Value Loss: {avg_val_loss:.4f}")


    # Save Model
    if USE_WANDB:
        wandb.save("checkpoints/go_resnet_final.pt")
        wandb.finish()

    torch.save(model.state_dict(), "checkpoints/go_resnet_final.pt")
    print("Model saved to checkpoints/go_resnet_final.pt")


if __name__ == "__main__":
    main()