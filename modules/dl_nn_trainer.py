import torch
from torch import nn
import os
from . import utils

class DLNNTrainer:
    def __init__(self, config, run_num=0):
        self.config = config
        self.run_num = run_num
        self.model_path = "models/dl_model.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        self.model = nn.Sequential(
            nn.Linear(5, self.config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(self.config["hidden_size"], self.config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(self.config["hidden_size"], 1),
            nn.Sigmoid()
        ).to(self.device)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"Loaded existing DL model from {self.model_path}")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        self.criterion = nn.MSELoss()

    def train_models(self):
        data = utils.fetch_data(run_num=self.run_num)
        inputs = torch.tensor(data, dtype=torch.float32).to(self.device)
        targets = torch.roll(inputs[:, 3], -1, dims=0)
        targets = targets.unsqueeze(1).to(self.device)  # Ensure targets are on GPU
        for epoch in range(self.config["epochs"]):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Saved DL model to {self.model_path}")
