
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


class SquareIntersectionPredictor:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def featurize(self, dxs, dys, theta2):

        r = torch.sqrt(dxs*dxs + dys*dys)

        return torch.stack([
            dxs,
            dys,
            r,
            torch.sin(theta2),
            torch.cos(theta2)
        ], axis=1).to(torch.float32)

    @torch.no_grad()
    def predict(self, dxs, dys, theta2):
        X = self.featurize(dxs, dys, theta2)
        return self.model(X).clamp(0, 1).reshape(-1)
