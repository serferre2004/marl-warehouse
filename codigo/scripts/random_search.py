#!/usr/bin/env python3
"""
Random Search de hiperparámetros para IA2C + RWARE/EPyMARL.

El script:
1. Genera configuraciones aleatorias reproducibles.
2. Crea un YAML temporal para cada trial.
3. Ejecuta scripts/run_experiment.py.
4. Busca los resultados del trial.
5. Extrae métricas agregadas y, cuando están disponibles, métricas de
   evaluación por intervalo.
6. Guarda una fila detallada por trial en un CSV.
7. Conserva el YAML utilizado para cada trial.

Uso desde la raíz del proyecto:
    python scripts/random_search.py --trials 30 --timesteps 300000 --seed 3

Opcional:
    python scripts/random_search.py --trials 30 --timesteps 300000 \
        --seed 3 --search-seed 12345 --keep-failed
"""

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_experiment.py"

SEARCH_DIR = PROJECT_ROOT / "experiments" / "random_search"
CONFIG_DIR = SEARCH_DIR / "configs"
RESULTS_CSV = SEARCH_DIR / "random_search_results.csv"


# ---------------------------------------------------------------------------
# Espacio de búsqueda
# ---------------------------------------------------------------------------

def sample_log_uniform(rng, low, high):
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_config(rng):
    """Genera una configuración IA2C."""

    return {
        # Optimización
        "lr": sample_log_uniform(rng, 1e-4, 1e-3),
        "gamma": rng.choice([0.95, 0.97, 0.99, 0.995]),
        "grad_norm_clip": rng.choice([5, 10, 20]),

        # Arquitectura
        "hidden_dim": rng.choice([64, 128, 256]),

        # Entrenamiento
        "batch_size": rng.choice([16, 32, 64]),

        # Exploración / política
        # Si tu configuración IA2C usa este parámetro, se conserva.
        "entropy_coef": sample_log_uniform(rng, 1e-3, 5e-2),

        # Estos valores se mantienen fijos en esta primera búsqueda.
        "optim_alpha": 0.99,
        "optim_eps": 1e-5,
        "add_value_last_step": True,
    }


# ---------------------------------------------------------------------------
# Utilidades de resultados
# ---------------------------------------------------------------------------

def flatten_dict(d, prefix=""):
    out = {}
    if not isinstance(d, dict):
        return out

    for key, value in d.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, name))
        elif isinstance(value, (list, tuple)):
            out[name] = json.dumps(value)
        else:
            out[name] = value
    return out


def numeric_values(values):
    out = []
    for value in values:
        try:
            if value is None or value == "":
                continue
            out.append(float(value))
        except (TypeError, ValueError):
            pass
    return out


def summarize(values):
    values = numeric_values(values)
    if not values:
        return {}

    values_sorted = sorted(values)
    n = len(values_sorted)

    def percentile(p):
        if n == 1:
            return values_sorted[0]
        pos = (n - 1) * p
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return values_sorted[lo]
        return values_sorted[lo] + (values_sorted[hi] - values_sorted[lo]) * (pos - lo)

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n

    return {
        "count": n,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": values_sorted[0],
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "max": values_sorted[-1],
    }


def find_json_files(root):
    return sorted(root.rglob("*.json")) if root.exists() else []


def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_metrics_from_json(obj, prefix=""):
    """
    Extrae recursivamente pares clave/valor numéricos.
    Para series/listas numéricas conserva estadísticas agregadas.
    """
    rows = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)

            if isinstance(value, dict):
                rows.update(extract_metrics_from_json(value, name))

            elif isinstance(value, list):
                nums = numeric_values(value)
                if nums:
                    s = summarize(nums)
                    for stat, val in s.items():
                        rows[f"{name}__{stat}"] = val

            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                rows[name] = value

    return rows


def collect_result_information(search_root, trial_start_time):
    """
    Busca JSON producidos por EPyMARL y devuelve:
      - archivos encontrados
      - métricas numéricas agregadas
      - algunas series/metricas reconocibles
    """
    files = find_json_files(search_root)

    # Preferir archivos modificados durante este trial.
    recent = []
    for path in files:
        try:
            if path.stat().st_mtime >= trial_start_time - 5:
                recent.append(path)
        except OSError:
            pass

    candidates = recent or files

    merged = {}
    selected_files = []

    # Métricas que queremos identificar explícitamente si existen.
    aliases = {
        "test_return_mean": [
            "test_return_mean",
            "return_mean",
            "test_return",
        ],
        "episode_length_mean": [
            "ep_length_mean",
            "episode_length_mean",
        ],
        "test_ep_length_mean": [
            "test_ep_length_mean",
        ],
    }

    explicit = {}

    for path in candidates:
        obj = load_json(path)
        if obj is None:
            continue

        selected_files.append(str(path.relative_to(PROJECT_ROOT)))

        flat = extract_metrics_from_json(obj)

        # Conservamos las métricas numéricas con un prefijo basado en el archivo.
        stem = path.stem
        for key, value in flat.items():
            merged[f"{stem}__{key}"] = value

        # Buscar aliases de forma flexible.
        lower_map = {k.lower(): v for k, v in flat.items()}
        for target, possible in aliases.items():
            for candidate in possible:
                for key, value in lower_map.items():
                    if key.endswith(candidate.lower()):
                        explicit[target] = value
                        break
                if target in explicit:
                    break

    # Evitar que el CSV se vuelva inmanejable por duplicación accidental.
    return {
        "result_files": ";".join(selected_files),
        "metrics": merged,
        "explicit": explicit,
    }


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------

def build_yaml(base_yaml_path, trial_id, hp, timesteps, seed, time_limit):
    with base_yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["name"] = f"ia2c_random_trial_{trial_id:03d}"
    config["algorithm"] = "ia2c"
    config["seed"] = seed
    config["time_limit"] = time_limit

    config.setdefault("training", {})
    config["training"]["timesteps"] = timesteps

    for key, value in hp.items():
        config[key] = value

    return config


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def append_csv(row):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    existing_fields = []
    if RESULTS_CSV.exists():
        with RESULTS_CSV.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                existing_fields = next(reader)
            except StopIteration:
                pass

    fields = list(existing_fields)
    for key in row:
        if key not in fields:
            fields.append(key)

    # Reescribir si aparecen nuevas columnas. Esto permite que trials con
    # distintos archivos de métricas no rompan el CSV.
    old_rows = []
    if RESULTS_CSV.exists():
        with RESULTS_CSV.open("r", encoding="utf-8", newline="") as f:
            old_rows = list(csv.DictReader(f))

    old_rows.append({k: row.get(k, "") for k in fields})

    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(old_rows)


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--search-seed", type=int, default=12345)
    parser.add_argument("--time-limit", type=int, default=2000)

    parser.add_argument(
        "--base-yaml",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "configs" / "ia2c_tiny_seed3.yaml",
    )

    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Conserva los YAML aunque el entrenamiento falle.",
    )

    args = parser.parse_args()

    if not RUNNER.exists():
        raise FileNotFoundError(f"No existe: {RUNNER}")

    if not args.base_yaml.exists():
        raise FileNotFoundError(f"No existe el YAML base: {args.base_yaml}")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.search_seed)

    print("=" * 72)
    print("IA2C RANDOM SEARCH")
    print("=" * 72)
    print(f"Trials:             {args.trials}")
    print(f"Timesteps / trial:  {args.timesteps:,}")
    print(f"Environment seed:   {args.seed}")
    print(f"Search seed:        {args.search_seed}")
    print(f"Time limit:         {args.time_limit}")
    print(f"Base YAML:          {args.base_yaml}")
    print(f"Results CSV:        {RESULTS_CSV}")
    print("=" * 72)

    for trial in range(1, args.trials + 1):
        hp = sample_config(rng)

        config = build_yaml(
            args.base_yaml,
            trial,
            hp,
            args.timesteps,
            args.seed,
            args.time_limit,
        )

        trial_yaml = CONFIG_DIR / f"trial_{trial:03d}.yaml"

        with trial_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

        started = time.time()
        start_dt = datetime.now().isoformat(timespec="seconds")

        print()
        print("-" * 72)
        print(f"TRIAL {trial}/{args.trials}")
        print("-" * 72)
        for key, value in hp.items():
            print(f"{key:22s}: {value}")

        command = [
            sys.executable,
            str(RUNNER),
            "--config",
            str(trial_yaml),
        ]

        return_code = None
        error_message = ""

        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=False,
            )
            return_code = completed.returncode

        except Exception as exc:
            error_message = repr(exc)

        finished = time.time()
        elapsed = finished - started

        result_info = collect_result_information(
            PROJECT_ROOT / "experiments" / "results",
            started,
        )

        row = {
            "trial": trial,
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "start_time": start_dt,
            "duration_seconds": round(elapsed, 3),
            "duration_minutes": round(elapsed / 60, 3),
            "environment_seed": args.seed,
            "search_seed": args.search_seed,
            "timesteps": args.timesteps,
            "time_limit": args.time_limit,
            "config_file": str(trial_yaml.relative_to(PROJECT_ROOT)),
            "error": error_message,
        }

        # Hiperparámetros explícitos.
        for key, value in hp.items():
            row[f"hp_{key}"] = value

        # Métricas reconocibles.
        for key, value in result_info["explicit"].items():
            row[key] = value

        # Todas las métricas numéricas encontradas.
        for key, value in result_info["metrics"].items():
            row[f"metric_{key}"] = value

        row["result_files"] = result_info["result_files"]

        append_csv(row)

        print(f"\nTrial {trial} terminado.")
        print(f"Return code: {return_code}")
        print(f"Duración: {elapsed / 60:.2f} min")
        print(f"CSV: {RESULTS_CSV}")

        if return_code != 0 and not args.keep_failed:
            try:
                trial_yaml.unlink()
            except OSError:
                pass

    print()
    print("=" * 72)
    print("RANDOM SEARCH TERMINADO")
    print("=" * 72)
    print(f"Resultados: {RESULTS_CSV}")
    print(f"Configuraciones: {CONFIG_DIR}")


if __name__ == "__main__":
    main()
