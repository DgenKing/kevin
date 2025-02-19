import torch
from . import utils

class DLNNStrategyGenerator:
    def __init__(self, config, run_num=0):
        self.config = config
        self.run_num = run_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "models/dl_model.pth"  # Define model_path
        self.model = torch.nn.Sequential(
            torch.nn.Linear(5, config["hidden_size"]),
            torch.nn.ReLU(),
            torch.nn.Linear(config["hidden_size"], config["hidden_size"]),
            torch.nn.ReLU(),
            torch.nn.Linear(config["hidden_size"], 1),
            torch.nn.Sigmoid()
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def generate_strategy(self):
        data = utils.fetch_data(run_num=self.run_num)
        inputs = torch.tensor(data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(inputs).cpu().numpy()
        threshold = float(predictions.mean())
        return {"type": "dl_nn_price", "threshold": threshold, "risk_percent": 0.5}
