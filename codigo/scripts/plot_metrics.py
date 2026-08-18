#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt

RELEVANT = ["test_return_mean", "return_mean", "critic_grad_norm", "episode_collisions_mean"]

def series(data, name):
    m = data.get(name)
    if not isinstance(m, dict): return None
    x, y = m.get("steps", []), m.get("values", [])
    n = min(len(x), len(y))
    return (x[:n], y[:n]) if n else None

def std_series(data, mean_name, n):
    """Busca la métrica de std correspondiente (mean -> std) y la recorta a n elementos."""
    if "mean" not in mean_name:
        return None
    std_name = mean_name.replace("mean", "std")
    s = series(data, std_name)
    if not s: return None
    _, y_std = s
    if len(y_std) < n: return None
    return y_std[:n]

def plot_with_band(ax, x, y, data, name, **kwargs):
    """Dibuja la curva y, si hay std disponible, la banda sombreada."""
    line, = ax.plot(x, y, **kwargs)
    y_std = std_series(data, name, len(x))
    if y_std:
        lo = [yi - si for yi, si in zip(y, y_std)]
        hi = [yi + si for yi, si in zip(y, y_std)]
        ax.fill_between(x, lo, hi, alpha=0.2, color=line.get_color())
    return line

def main():
    p = argparse.ArgumentParser(description="Grafica métricas relevantes de metrics.json de EPyMARL")
    p.add_argument("metrics", nargs="?", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    a = p.parse_args()
    path = a.metrics
    if path is None:
        found = list(Path.cwd().rglob("metrics.json"))
        if not found: raise FileNotFoundError("No se encontró metrics.json")
        if len(found) == 1: path = found[0]
        else:
            for i, f in enumerate(found, 1): print(f"{i}. {f}")
            path = found[int(input("Selecciona: "))-1]
    if not path.exists(): raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f: data = json.load(f)
    out = a.out or path.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    found = [(n, *series(data,n)) for n in RELEVANT if series(data,n)]
    if not found:
        print("No se encontraron métricas conocidas. Disponibles:", *data.keys(), sep="\n- "); return
    for name, x, y in found:
        fig, ax = plt.subplots(figsize=(10,6))
        plot_with_band(ax, x, y, data, name)
        ax.set(title=name, xlabel="Pasos de entrenamiento", ylabel=name); ax.grid(True, alpha=.3); fig.tight_layout(); fig.savefig(out/f"{name}.png", dpi=150); plt.close(fig)
    tr, te = series(data,"train_return_mean"), series(data,"test_return_mean")
    if tr or te:
        fig, ax = plt.subplots(figsize=(10,6))
        if tr: plot_with_band(ax, *tr, data, "train_return_mean", label="Train return mean")
        if te: plot_with_band(ax, *te, data, "test_return_mean", label="Test return mean")
        ax.set(title="Recompensa durante el entrenamiento", xlabel="Pasos de entrenamiento", ylabel="Recompensa"); ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out/"learning_curve.png", dpi=150); plt.close(fig)
    print(f"Gráficas guardadas en: {out}")
    if a.show: plt.show()

if __name__ == "__main__": main()