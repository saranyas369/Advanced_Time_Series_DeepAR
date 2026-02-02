import matplotlib.pyplot as plt
import numpy as np

def plot_forecast_vs_actual(y_true, y_pred):
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label="Actual", linewidth=2)
    plt.plot(y_pred, label="Forecast", linestyle="--")
    plt.title("Forecast vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_uncertainty(y_true, mu, sigma):
    t = np.arange(len(y_true))
    plt.figure(figsize=(10,5))
    plt.plot(t, y_true, label="Actual")
    plt.plot(t, mu, label="Mean Forecast", linestyle="--")
    plt.fill_between(t, mu-2*sigma, mu+2*sigma, alpha=0.3, label="Uncertainty")
    plt.legend()
    plt.tight_layout()
    plt.show()