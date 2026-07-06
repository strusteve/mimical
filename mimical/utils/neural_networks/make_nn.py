from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np

install_dir = os.path.dirname(os.path.realpath(__file__))

device = (torch.accelerator.current_accelerator()
          if torch.accelerator.is_available()
          else torch.device("cpu"))


def unit_square():
    return Polygon([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])


def transformed_square(center, theta):
    poly = unit_square()
    poly = rotate(poly, theta, origin=(0, 0), use_radians=True)
    poly = translate(poly, xoff=center[0], yoff=center[1])
    return poly


def intersection_area(dx, dy, theta2):
    A = transformed_square((0, 0), 0)
    B = transformed_square((dx, dy), theta2)
    return A.intersection(B).area


def generate_dataset(n):
    dxs = np.random.normal(0, 0.5, n)
    dys = np.random.normal(0, 0.5, n)
    t2 = np.random.uniform(0, 2*np.pi, n)

    r = np.sqrt(dxs*dxs + dys*dys)

    X = np.stack([dxs,
                  dys,
                  r,
                  np.sin(t2),
                  np.cos(t2)], axis=1)

    y = np.array([intersection_area(dx, dy, th2)
                  for dx, dy, th2
                  in zip(dxs, dys, t2)], dtype=np.float32).reshape(-1, 1)

    return X, y


def make_nn(MLP, device="cpu"):

    print('Training square intersection model...')
    # data
    X, y = generate_dataset(1_000_000)
    # Shuffle indices
    perm = np.random.permutation(len(X))

    # 90% train, 10% validation
    split = int(0.9 * len(X))
    train_idx = perm[:split]
    val_idx = perm[split:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32, device=device)

    X_val = torch.tensor(X[val_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=512,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=512,
        shuffle=False
    )

    model = MLP.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(50):

        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1:2d}: train={train_loss:.6f}  "
              "val={val_loss:.6f}")

    torch.save(model.state_dict(), install_dir + "/square_intersection_nn.pth")

    model.eval()
    pred = model(X_val).cpu().detach().numpy().flatten()
    true = y_val.cpu().numpy().flatten()
    err = pred - true
    print(np.mean(np.abs(err)))
    print(np.max(np.abs(err)))
    print(np.percentile(np.abs(err), 99))

    return model
