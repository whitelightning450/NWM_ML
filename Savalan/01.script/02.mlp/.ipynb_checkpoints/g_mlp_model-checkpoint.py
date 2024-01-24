# This file created on 01/13/2024 by savalan

# Import packages ==============================
# main packages
import torch
import torch.nn as nn
import torch.optim as optim

# Functions ==============================

class CustomMLP(nn.Module):
    def __init__(self, layer_sizes, device=None):
        super(CustomMLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.loss_function = nn.MSELoss()
        self.device = device
        self.to(self.device)
        self.validation_indicator = 0

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x

    def train_model(self, train_loader, epochs, optimizer, early_stopping_patience=0, save_path=None, val_loader=None):
        best_val_loss = float('inf')
        epochs_no_improve = 0


        for epoch in range(epochs):
            self.train()  # Set the model to training mode
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

            
            val_loss = 0
            best_model_parameters = self.state_dict()
            if val_loader is not None:
                self.validation_indicator = 1
                val_loss = self.evaluate_model(val_loader)[1]

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_parameters = self.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve == early_stopping_patience and early_stopping_patience > 0:
                    print('Early stopping triggered')
                    break
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}', f'Validation Loss: {val_loss}')
        self.validation_indicator = 0
        return best_model_parameters
        print('Training is done!')

    def evaluate_model(self, data_loader):
        self.eval()  # Set the model to evaluation mode
        total_loss = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.forward(inputs)
                loss = self.loss_function(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)
        avg_loss = total_loss / total
        if self.validation_indicator == 0:
            print(f'Validation Loss: {avg_loss}')
        return outputs, avg_loss
        #outputs if self.validation_indicator == 0 else avg_loss

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path, map_location=self.device))
# early stop and checkpoint shoudl be different functions. 