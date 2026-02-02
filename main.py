import torch, json
from data_loader import load_multivariate_data
from deepar_prob import DeepAR
from plots import plot_forecast_vs_actual, plot_uncertainty

with open("config.json") as f:
    cfg = json.load(f)

df = load_multivariate_data()
features = ["target","temp","promotion","lag_1","lag_7","holiday"]
X = torch.tensor(df[features].values, dtype=torch.float32).unsqueeze(0)

model = DeepAR(input_size=len(features))
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

for e in range(cfg["epochs"]):
    optimizer.zero_grad()
    mu, sigma = model(X)
    loss = ((mu - X[:,:,0:1])**2 / (2*sigma**2) + torch.log(sigma)).mean()
    loss.backward()
    optimizer.step()

print("Training complete")

y_true = X[0,:,0].detach().numpy()
mu_np = mu[0,:,0].detach().numpy()
sigma_np = sigma[0,:,0].detach().numpy()

plot_forecast_vs_actual(y_true, mu_np)
plot_uncertainty(y_true, mu_np, sigma_np)