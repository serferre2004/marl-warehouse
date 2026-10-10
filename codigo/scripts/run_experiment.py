import argparse
from html import parser
from logging import config
from datetime import datetime
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def resolve_sacred_run_id(config, results_path, environment_key):
    algorithm_config_path = (
        PROJECT_ROOT
        / "epymarl"
        / "src"
        / "config"
        / "algs"
        / f"{config['algorithm']}.yaml"
    )
    algorithm_config = load_config(algorithm_config_path)
    experiment_name = algorithm_config.get("name", config["algorithm"])
    sacred_base_dir = (
        results_path
        / "sacred"
        / experiment_name
        / environment_key.replace(":", "_")
    )

    seed = config["seed"]
    run_id = f"seed{seed}"
    run_dir = sacred_base_dir / run_id
    if not run_dir.exists():
        return run_id

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_id = f"seed{seed}_{timestamp}"
    run_dir = sacred_base_dir / run_id
    suffix = 2
    while run_dir.exists():
        run_id = f"seed{seed}_{timestamp}_{suffix}"
        run_dir = sacred_base_dir / run_id
        suffix += 1

    return run_id

def main():
    parser = argparse.ArgumentParser(
    description="Ejecuta un único experimento de EPyMARL."
    )


    parser.add_argument(
    "config",
    type=Path,
    help="Ruta al archivo YAML de configuración.",
    )

    args = parser.parse_args()

    config_path = args.config.resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración: {config_path}"
        )

    config = load_config(config_path)

    print(f"Ejecutando experimento: {config['name']}")
    print(f"Algoritmo: {config['algorithm']}")
    print(f"Escenario: {config['environment']['scenario']}")
    print(f"Número de agentes: {config['environment']['n_agents']}")
    print(f"Semilla: {config['seed']}")

    epymarl_dir = PROJECT_ROOT / "epymarl"

    results_path = PROJECT_ROOT / config["results"]["path"]
    results_path.mkdir(parents=True, exist_ok=True)

    key = "-".join([config['environment']['key'], config['environment']['scenario'], f"{config['environment']['n_agents']}ag", "v2"])
    sacred_run_id = resolve_sacred_run_id(config, results_path, key)
    print(f"ID de Sacred: {sacred_run_id}")


    command = [
        sys.executable,
        "src/main.py",
        f"--config={config['algorithm']}",
        "--env-config=gymma",
        f"--id={sacred_run_id}",
        "with",

        f"common_reward={str(config['reward']['common']).capitalize()}",

        # Environment
        f"env_args.key={key}",
        f"env_args.time_limit={config['time_limit']}",

        f"env_args.request_queue_size={config['environment']['request_queue_size']}",
        f"env_args.sensor_range={config['environment']['sensor_range']}",
        f"env_args.msg_bits={config['environment']['msg_bits']}",
        f"env_args.reward_type={config['environment']['reward_type']}",
        f"env_args.observation_type={config['environment']['observation_type']}",
        f"env_args.layout={config['environment']['layout']}",

        f"seed={config['seed']}",

        # Training
        f"t_max={config['training']['timesteps']}",
        f"reward_scalarisation={config['reward']['scalarisation']}",

        # Logging
        f"runner_log_interval={config['logging']['runner_interval']}",
        f"learner_log_interval={config['logging']['learner_interval']}",
        f"log_interval={config['logging']['general_interval']}",

        # Checkpoints

        f"save_model={str(config['checkpoint']['save']).lower()}",
        f"save_model_interval={config['checkpoint']['interval']}",

        # Evaluation
        f"evaluate={str(config['evaluation']['enabled']).lower()}",
        f"test_interval={config['evaluation']['interval']}",
        f"test_nepisode={config['evaluation']['episodes']}",

        # Results
        f"local_results_path={results_path}",
    ]

    keys_setted = ["algorithm", "environment", "training", "logging", "checkpoint", "evaluation", "reward", "results", "seed", "time_limit", "name"]
    for key in keys_setted:
        del config[key]

    for key, value in config.items():
        if value is None: continue
        if isinstance(value, bool):
            value = str(value).lower()
        command.append(f"{key}={value}")
    
    env = os.environ.copy()
    env["EPYMARL_RESULTS_PATH"] = str(results_path)

    subprocess.run(
        command,
        cwd=epymarl_dir,
        env=env,
        check=True,
    )

if __name__ == "__main__":
    main()
