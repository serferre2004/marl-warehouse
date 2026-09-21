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
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_experiment.py"
SUPPORTED_ALGORITHMS = ("ia2c", "ippo", "maa2c", "mappo")


def resolve_search_paths(algorithm):
    algorithm = (algorithm or "ia2c").lower()
    search_dir = PROJECT_ROOT / "experiments" / "random_search" / algorithm
    return {
        "search_dir": search_dir,
        "config_dir": search_dir / "configs",
        "search_results_dir": search_dir / "results",
        "results_csv": search_dir / "random_search_results.csv",
    }


# ---------------------------------------------------------------------------
# Espacio de búsqueda
# ---------------------------------------------------------------------------

def sample_log_uniform(rng, low, high):
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_config(rng, algorithm):
    """Genera una configuración aleatoria para el algoritmo objetivo."""

    algorithm = (algorithm or "ia2c").lower()

    if algorithm == "ippo":
        return {
            # Optimización
            "lr": sample_log_uniform(rng, 1e-4, 3e-3),
            "gamma": rng.choice([0.95, 0.97, 0.99, 0.995]),
            "grad_norm_clip": rng.choice([5, 10, 20]),

            # Arquitectura
            "hidden_dim": rng.choice([64, 128, 256]),

            # Entrenamiento
            "batch_size": rng.choice([8, 16, 32]),
            "batch_size_run": rng.choice([8, 16, 32]),
            "epochs": rng.choice([2, 4, 6]),
            "eps_clip": rng.choice([0.1, 0.2, 0.3]),
            "target_update_interval_or_tau": rng.choice([0.01, 0.05, 0.1]),

            # Exploración / política
            "entropy_coef": sample_log_uniform(rng, 1e-3, 5e-2),

            # Valores fijos para mantener consistencia con las configs base.
            "optim_alpha": 0.99,
            "optim_eps": 1e-5,
            "add_value_last_step": True,
        }

    if algorithm == "mappo":
        batch_size = rng.choice([8, 10, 16, 32])
        batch_size_run = rng.choice([8, 10, 16, 32])

        return {
            # Optimizacion
            "lr": sample_log_uniform(rng, 1e-4, 3e-3),
            "gamma": rng.choice([0.95, 0.97, 0.99, 0.995]),
            "grad_norm_clip": rng.choice([5, 10, 20]),

            # Arquitectura
            "hidden_dim": rng.choice([64, 128, 256]),

            # Entrenamiento PPO con critic centralizado segun mappo.yaml.
            "batch_size": batch_size,
            "batch_size_run": batch_size_run,
            "buffer_size": max(10, batch_size),
            "epochs": rng.choice([2, 4, 6]),
            "eps_clip": rng.choice([0.1, 0.2, 0.3]),
            "target_update_interval_or_tau": rng.choice([0.01, 0.05, 0.1]),

            # Exploracion / politica
            "entropy_coef": sample_log_uniform(rng, 1e-4, 1e-2),

            # Valores fijos para mantener consistencia con mappo.yaml.
            "action_selector": "soft_policies",
            "mask_before_softmax": True,
            "runner": "parallel",
            "obs_agent_id": True,
            "obs_last_action": False,
            "obs_individual_obs": False,
            "agent_output_type": "pi_logits",
            "learner": "ppo_learner",
            "use_rnn": True,
            "q_nstep": 5,
            "critic_type": "cv_critic",
            "optim_alpha": 0.99,
            "optim_eps": 1e-5,
            "add_value_last_step": True,
            "standardise_returns": False,
            "standardise_rewards": True,
        }

    if algorithm == "maa2c":
        return {
            # Optimización
            "lr": sample_log_uniform(rng, 1e-4, 1e-3),
            "gamma": rng.choice([0.95, 0.97, 0.99, 0.995]),
            "grad_norm_clip": rng.choice([5, 10, 20]),

            # Arquitectura
            "hidden_dim": rng.choice([64, 128, 256]),

            # Entrenamiento
            "batch_size": rng.choice([8, 16, 32]),
            "batch_size_run": rng.choice([8, 16, 32]),
            "target_update_interval_or_tau": rng.choice([0.01, 0.05, 0.1]),

            # Exploración / política
            "entropy_coef": sample_log_uniform(rng, 1e-3, 5e-2),

            # Valores fijos para mantener consistencia con las configs base.
            "optim_alpha": 0.99,
            "optim_eps": 1e-5,
            "add_value_last_step": True,
        }

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
    files = sorted(search_root.rglob("metrics.json")) if search_root.exists() else []

    recent = []
    for path in files:
        try:
            if path.stat().st_mtime >= trial_start_time - 5:
                recent.append(path)
        except OSError:
            pass

    candidates = recent or files
    if not candidates:
        return {
            "result_files": "",
            "metrics": {},
        }

    metrics_path = max(candidates, key=lambda path: path.stat().st_mtime)
    obj = load_json(metrics_path)
    if not isinstance(obj, dict):
        return {
            "result_files": str(metrics_path.relative_to(PROJECT_ROOT)),
            "metrics": {},
        }

    metric = obj.get("test_return_mean", {})
    values = numeric_values(metric.get("values", [])) if isinstance(metric, dict) else []
    steps = metric.get("steps", []) if isinstance(metric, dict) else []

    metrics = {}
    summary = summarize(values)
    if summary:
        max_idx = max(range(len(values)), key=lambda idx: values[idx])
        metrics = {
            "test_return_mean_max": summary["max"],
            "test_return_mean_mean": summary["mean"],
            "test_return_mean_std": summary["std"],
            "test_return_mean_count": summary["count"],
            "test_return_mean_last": values[-1],
        }

        if max_idx < len(steps):
            metrics["test_return_mean_max_step"] = steps[max_idx]

    return {
        "result_files": str(metrics_path.relative_to(PROJECT_ROOT)),
        "metrics": metrics,
    }


def latest_metrics_snapshot(search_root, trial_start_time):
    files = sorted(search_root.rglob("metrics.json")) if search_root.exists() else []

    recent = []
    for path in files:
        try:
            if path.stat().st_mtime >= trial_start_time - 5:
                recent.append(path)
        except OSError:
            pass

    candidates = recent or files
    if not candidates:
        return {}

    metrics_path = max(candidates, key=lambda path: path.stat().st_mtime)
    obj = load_json(metrics_path)
    if not isinstance(obj, dict):
        return {}

    metric = obj.get("test_return_mean", {})
    if not isinstance(metric, dict):
        return {}

    steps = metric.get("steps", [])
    values = numeric_values(metric.get("values", []))

    if not steps and not values:
        return {}

    return {
        "metrics_path": metrics_path,
        "steps": steps,
        "values": values,
        "last_step": steps[-1] if steps else None,
        "last_value": values[-1] if values else None,
        "max_value": max(values) if values else None,
    }


def should_trigger_early_stop(search_root, trial_start_time, early_stop_steps, early_stop_threshold):
    if early_stop_steps is None or early_stop_steps <= 0:
        return False

    if early_stop_threshold is None:
        return False

    snapshot = latest_metrics_snapshot(search_root, trial_start_time)
    if not snapshot:
        return False

    last_step = snapshot.get("last_step")
    max_value = snapshot.get("max_value")

    if last_step is None or max_value is None:
        return False

    if last_step < early_stop_steps:
        return False

    return max_value < early_stop_threshold


def stop_trial_process(process):
    if process.poll() is not None:
        return process.returncode

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return process.returncode

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        process.terminate()

    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    return process.returncode

# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------

def load_algorithm_default_config(algorithm):
    algorithm = (algorithm or "ia2c").lower()
    default_config_path = (
        PROJECT_ROOT
        / "epymarl"
        / "src"
        / "config"
        / "algs"
        / f"{algorithm}.yaml"
    )

    if not default_config_path.exists():
        return {}

    with default_config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def legacy_algorithm_keys_for(algorithm):
    algorithm = (algorithm or "ia2c").lower()
    defaults = {name: load_algorithm_default_config(name) for name in SUPPORTED_ALGORITHMS}

    union_keys = set()
    for cfg in defaults.values():
        union_keys.update(cfg.keys())

    shared_keys = set.intersection(*(set(cfg.keys()) for cfg in defaults.values()))
    current_keys = set(defaults.get(algorithm, {}).keys())

    return union_keys - shared_keys - current_keys


def prune_legacy_algorithm_keys(base_config, algorithm):
    if not isinstance(base_config, dict):
        return base_config

    legacy_keys = legacy_algorithm_keys_for(algorithm)
    if not legacy_keys:
        return base_config

    pruned_config = dict(base_config)
    for key in list(pruned_config.keys()):
        if key in legacy_keys:
            pruned_config.pop(key, None)

    return pruned_config


def merge_algorithm_defaults(base_config, algorithm):
    algorithm_defaults = load_algorithm_default_config(algorithm)
    if not algorithm_defaults:
        return base_config

    for key, value in algorithm_defaults.items():
        if key in {"name", "t_max"}:
            continue
        if key not in base_config:
            base_config[key] = value

    return base_config


def build_yaml(
    base_yaml_path,
    algorithm,
    trial_id,
    hp,
    timesteps,
    seed,
    time_limit,
    trial_results_dir,
):
    with base_yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    algorithm = (algorithm or "ia2c").lower()
    config = prune_legacy_algorithm_keys(config, algorithm)
    config = merge_algorithm_defaults(config, algorithm)

    config["name"] = f"{algorithm}_random_trial_{trial_id:03d}"
    config["algorithm"] = algorithm
    config["seed"] = seed
    config["time_limit"] = time_limit

    config.setdefault("training", {})
    config["training"]["timesteps"] = timesteps
    config.setdefault("results", {})
    config["results"]["path"] = str(trial_results_dir.relative_to(PROJECT_ROOT))

    for key, value in hp.items():
        config[key] = value

    return config


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def append_csv(row, results_csv):
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    existing_fields = []
    if results_csv.exists():
        with results_csv.open("r", encoding="utf-8", newline="") as f:
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
    if results_csv.exists():
        with results_csv.open("r", encoding="utf-8", newline="") as f:
            old_rows = list(csv.DictReader(f))

    old_rows.append({k: row.get(k, "") for k in fields})

    with results_csv.open("w", encoding="utf-8", newline="") as f:
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
    parser.add_argument("--time-limit", type=int, default=500)
    parser.add_argument(
        "--algorithm",
        choices=list(SUPPORTED_ALGORITHMS),
        help="Algoritmo objetivo para la búsqueda. Si no se indica, se infiere desde el YAML base.",
    )

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
    parser.add_argument(
        "--early-stop-steps",
        type=int,
        default=10_000_000,
        help="Si el valor máximo de test_return_mean aún no supera el umbral para esta cantidad de pasos, aborta el trial antes de gastar más cálculo.",
    )
    parser.add_argument(
        "--early-stop-threshold",
        type=float,
        default=1.0,
        help="Umbral de retorno medio en test que activa el early stopping.",
    )
    parser.add_argument(
        "--early-stop-check-interval",
        type=float,
        default=30.0,
        help="Segundos entre comprobaciones del early stopping.",
    )

    args = parser.parse_args()

    if not RUNNER.exists():
        raise FileNotFoundError(f"No existe: {RUNNER}")

    if not args.base_yaml.exists():
        raise FileNotFoundError(f"No existe el YAML base: {args.base_yaml}")

    with args.base_yaml.open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f) or {}

    algorithm = (args.algorithm or base_config.get("algorithm") or "ia2c").lower()
    search_paths = resolve_search_paths(algorithm)

    search_paths["config_dir"].mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.search_seed)

    print("=" * 72)
    print(f"{algorithm.upper()} RANDOM SEARCH")
    print("=" * 72)
    print(f"Trials:             {args.trials}")
    print(f"Timesteps / trial:  {args.timesteps:,}")
    print(f"Environment seed:   {args.seed}")
    print(f"Search seed:        {args.search_seed}")
    print(f"Time limit:         {args.time_limit}")
    print(f"Algorithm:          {algorithm}")
    print(f"Base YAML:          {args.base_yaml}")
    print(f"Search dir:         {search_paths['search_dir']}")
    print(f"Results CSV:        {search_paths['results_csv']}")
    print("=" * 72)

    for trial in range(1, args.trials + 1):
        hp = sample_config(rng, algorithm)
        trial_results_dir = search_paths["search_results_dir"] / f"trial_{trial:03d}"

        config = build_yaml(
            args.base_yaml,
            algorithm,
            trial,
            hp,
            args.timesteps,
            args.seed,
            args.time_limit,
            trial_results_dir,
        )

        trial_yaml = search_paths["config_dir"] / f"trial_{trial:03d}.yaml"

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
            str(trial_yaml),
        ]

        return_code = None
        error_message = ""
        early_stopped = False

        try:
            popen_kwargs = {"cwd": PROJECT_ROOT}
            if os.name != "nt":
                popen_kwargs["start_new_session"] = True

            process = subprocess.Popen(
                command,
                **popen_kwargs,
            )

            while process.poll() is None:
                if should_trigger_early_stop(
                    trial_results_dir,
                    started,
                    args.early_stop_steps,
                    args.early_stop_threshold,
                ):
                    early_stopped = True
                    stop_trial_process(process)

                    print(
                        f"\nEarly stopping activado para trial {trial}: "
                        f"test_return_mean máximo aún < {args.early_stop_threshold} "
                        f"tras {args.early_stop_steps} pasos."
                    )
                    break

                time.sleep(args.early_stop_check_interval)

            if process.poll() is None:
                process.wait()

            return_code = process.returncode

        except Exception as exc:
            error_message = repr(exc)

        finished = time.time()
        elapsed = finished - started

        result_info = collect_result_information(
            trial_results_dir,
            started,
        )

        row = {
            "trial": trial,
            "status": "early_stopped" if early_stopped else ("completed" if return_code == 0 else "failed"),
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
            "early_stop_steps": args.early_stop_steps,
            "early_stop_threshold": args.early_stop_threshold,
        }

        # Hiperparámetros explícitos.
        for key, value in hp.items():
            row[f"hp_{key}"] = value

        # Todas las métricas numéricas encontradas.
        for key, value in result_info["metrics"].items():
            row[f"{key}"] = value

        row["result_files"] = result_info["result_files"]

        append_csv(row, search_paths["results_csv"])

        print(f"\nTrial {trial} terminado.")
        print(f"Return code: {return_code}")
        print(f"Estado: {row['status']}")
        print(f"Duración: {elapsed / 60:.2f} min")
        print(f"CSV: {search_paths['results_csv']}")

        if return_code != 0 and not args.keep_failed:
            try:
                trial_yaml.unlink()
            except OSError:
                pass

    print()
    print("=" * 72)
    print("RANDOM SEARCH TERMINADO")
    print("=" * 72)
    print(f"Resultados: {search_paths['results_csv']}")
    print(f"Configuraciones: {search_paths['config_dir']}")


if __name__ == "__main__":
    main()
