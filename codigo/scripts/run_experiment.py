import argparse
from html import parser
from logging import config
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

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


    command = [
        sys.executable,
        "src/main.py",
        f"--config={config['algorithm']}",
        "--env-config=gymma",
        "with",

        # Environment
        f"env_args.key={key}",
        f"env_args.time_limit={config['time_limit']}",

        f"env_args.request_queue_size={config['environment']['request_queue_size']}",
        f"env_args.sensor_range={config['environment']['sensor_range']}",
        f"env_args.msg_bits={config['environment']['msg_bits']}",
        f"env_args.reward_type={config['environment']['reward_type']}",
        f"env_args.observation_type={config['environment']['observation_type']}",
        f"env_args.max_steps={config['time_limit']}",       

        f"seed={config['seed']}",

        # Training
        f"runner={config['runner']}",
        f"t_max={config['training']['timesteps']}",

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

    command.extend([
        f"gamma={config['gamma']}",
        f"lr={config['lr']}",
        f"batch_size={config['batch_size']}",
        f"buffer_size={config['buffer_size']}",
        f"hidden_dim={config['hidden_dim']}",
        f"grad_norm_clip={config['grad_norm_clip']}",
        f"optim_alpha={config['optim_alpha']}",
        f"optim_eps={config['optim_eps']}",
        f"entropy_coef={config['entropy_coef']}",
        f"use_rnn={str(config['use_rnn']).lower()}",
        f"q_nstep={config['q_nstep']}",
        f"action_selector={config['action_selector']}",
        f"mask_before_softmax={str(config['mask_before_softmax']).lower()}",
        f"target_update_interval_or_tau={config['target_update_interval_or_tau']}",
        f"agent={config['agent']}",
        f"agent_output_type={config['agent_output_type']}",
        f"learner={config['learner']}",
        f"standardise_returns={str(config['standardise_returns']).lower()}",
        f"standardise_rewards={str(config['standardise_rewards']).lower()}",
        f"critic_type={config['critic_type']}"
    ])

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
