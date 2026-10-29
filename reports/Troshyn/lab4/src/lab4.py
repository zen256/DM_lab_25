import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

MATERNAL_CSV = "Maternal Health Risk Data Set.csv"
RICE_ARFF = "Rice_Cammeo_Osmancik.arff"

HIDDEN = [32, 32, 16]           # три скрытых слоя
PRE_EPOCHS = 100
HEAD_EPOCHS = 50                # обучение только последнего слоя
FT_EPOCHS = 200                 # обучение всей сети, одинаково для всех вариантов
LR, MOMENTUM, BATCH = 0.1, 0.9, 32
RBM_LR = 0.05
SEED = 0

class Layer(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_in, n_out))
        nn.init.xavier_uniform_(self.W)
        self.T = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        return x @ self.W - self.T

def build_net(sizes):
    layers = []
    for i, (n_in, n_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(Layer(n_in, n_out))
        if i < len(sizes) - 2:
            layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

def layers_of(net):
    return [m for m in net if isinstance(m, Layer)]

def loader(*tensors, gen):
    return DataLoader(TensorDataset(*tensors), batch_size=BATCH, shuffle=True, generator=gen)

#Предобучение: автоэнкодер
def train_autoencoder(X, n_hidden, gen):
    enc, dec = Layer(X.shape[1], n_hidden), Layer(n_hidden, X.shape[1])
    opt = torch.optim.SGD([*enc.parameters(), *dec.parameters()], lr=LR, momentum=MOMENTUM)
    for _ in range(PRE_EPOCHS):
        for (x,) in loader(X, gen=gen):
            recon = torch.sigmoid(dec(torch.sigmoid(enc(x))))
            loss = 0.5 * ((recon - x) ** 2).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return enc.W.data, enc.T.data


#Предобучение: RBM (алгоритм CD-1)
def train_rbm(X, n_hidden, gen):
    n_in = X.shape[1]
    W = 0.1 * torch.randn(n_in, n_hidden, generator=gen)
    Tv, Th = torch.zeros(n_in), torch.zeros(n_hidden)       # пороги видимых и скрытых нейронов
    vW, vTv, vTh = torch.zeros_like(W), torch.zeros_like(Tv), torch.zeros_like(Th)
    for epoch in range(PRE_EPOCHS):
        momentum = 0.5 if epoch < 5 else MOMENTUM
        for (v0,) in loader(X, gen=gen):
            h0 = torch.sigmoid(v0 @ W - Th)
            h0_sample = (torch.rand(h0.shape, generator=gen) < h0).float()
            v1 = torch.sigmoid(h0_sample @ W.T - Tv)                     # реконструкция x(1)
            h1 = torch.sigmoid(v1 @ W - Th)
            n = len(v0)
            vW = momentum * vW + RBM_LR * (v0.T @ h0 - v1.T @ h1) / n
            vTv = momentum * vTv - RBM_LR * (v0 - v1).mean(dim=0)
            vTh = momentum * vTh - RBM_LR * (h0 - h1).mean(dim=0)
            W, Tv, Th = W + vW, Tv + vTv, Th + vTh
    return W, Th

def pretrain(net, X, gen, train_layer):
    data = X
    for layer in layers_of(net)[:-1]:
        W, T = train_layer(data, layer.W.shape[1], gen)
        layer.W.data.copy_(W)
        layer.T.data.copy_(T)
        data = torch.sigmoid(data @ W - T)

def set_hidden_trainable(net, flag):
    for layer in layers_of(net)[:-1]:
        for p in layer.parameters():
            p.requires_grad = flag

def train_supervised(net, X, y, epochs, gen):
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for _ in range(epochs):
        total = 0.0
        for xb, yb in loader(X, y, gen=gen):
            loss = loss_fn(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        history.append(total / len(X))
    return history

def fit(net, X, y, gen, pretrained):
    history = []
    if pretrained:
        set_hidden_trainable(net, False)
        history += train_supervised(net, X, y, HEAD_EPOCHS, gen)
        set_hidden_trainable(net, True)
    history += train_supervised(net, X, y, FT_EPOCHS, gen)
    return history

def evaluate(net, X, y):
    with torch.no_grad():
        pred = net(X).argmax(dim=1).numpy()
    return accuracy_score(y, pred), f1_score(y, pred, average="macro"), confusion_matrix(y, pred)

def load_maternal():
    df = pd.read_csv(MATERNAL_CSV).drop_duplicates()
    classes = ["low risk", "mid risk", "high risk"]
    y = df["RiskLevel"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    return df.drop(columns="RiskLevel").to_numpy(float), y, classes

def load_rice():
    df = pd.DataFrame(arff.loadarff(RICE_ARFF)[0])
    df["Class"] = df["Class"].str.decode("utf-8")
    classes = ["Cammeo", "Osmancik"]
    y = df["Class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    return df.drop(columns="Class").to_numpy(float), y, classes

def experiment(name, X, y, classes):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    scaler = MinMaxScaler().fit(Xtr)                        # [0, 1] для сигмоид, AE и RBM
    Xtr = torch.tensor(scaler.transform(Xtr), dtype=torch.float32)
    Xte = torch.tensor(scaler.transform(Xte), dtype=torch.float32)
    Ytr = torch.tensor(ytr, dtype=torch.long)
    sizes = [X.shape[1], *HIDDEN, len(classes)]

    methods = {"Без предобучения": None, "Автоэнкодер": train_autoencoder, "RBM": train_rbm}
    histories = {}
    print(f"\n=== {name} ===")
    for title, train_layer in methods.items():
        torch.manual_seed(SEED)
        gen = torch.Generator().manual_seed(SEED)
        net = build_net(sizes)
        if train_layer:
            pretrain(net, Xtr, gen, train_layer)
        histories[title] = fit(net, Xtr, Ytr, gen, pretrained=train_layer is not None)
        acc, f1, cm = evaluate(net, Xte, yte)
        print(f"\n{title}: accuracy = {acc:.3f}, macro-F1 = {f1:.3f}")
        print(f"Матрица ошибок (строки: истина, столбцы: {', '.join(classes)}):\n{cm}")

    plt.figure()
    for title, history in histories.items():
        plt.plot(history, label=title)
    plt.title(name)
    plt.xlabel("Эпоха обучения с учителем")
    plt.ylabel("CE на обучающей выборке")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"curves_{name.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    plt.close()

def main():
    experiment("Maternal Health Risk", *load_maternal())
    experiment("Rice", *load_rice())


if __name__ == "__main__":
    main()
