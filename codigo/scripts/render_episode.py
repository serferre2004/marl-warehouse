import argparse
import os
import sys

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch as th

# Permite importar los módulos de EPyMARL desde scripts/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EPYMARL = os.path.join(ROOT, "epymarl")
SRC = os.path.join(EPYMARL, "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

from modules.agents.rnn_agent import RNNAgent
from rware.warehouse import RewardType


def get_agent_action_count(action_space):
    """Return the discrete action count from RWARE's per-agent action space."""
    agent_action_space = action_space[0]

    if isinstance(agent_action_space, gym.spaces.Discrete):
        return agent_action_space.n

    # Con comunicación (msg_bits > 0) el espacio es MultiDiscrete([5, 2, ...]);
    # la política opera sobre el producto cartesiano aplanado.
    if isinstance(agent_action_space, gym.spaces.MultiDiscrete):
        return int(np.prod(agent_action_space.nvec))

    raise ValueError(
        "render_episode.py espera acciones discretas por agente. "
        f"Espacio recibido: {agent_action_space}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Renderiza un episodio usando un checkpoint de EPyMARL."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Directorio que contiene agent.th.",
    )

    parser.add_argument(
        "--output",
        default="episode.mp4",
        help="Archivo de video de salida.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Semilla base del episodio. Cuando se especifica, se prueban "
            "--num-seeds semillas consecutivas a partir de este valor y "
            "solo se renderiza (guarda video) la que obtenga mejor "
            "recompensa total."
        ),
    )

    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help=(
            "Número de semillas a evaluar antes de renderizar. "
            "Usa un valor entre 10 y 30 para elegir el mejor episodio "
            "(por defecto 10). Se prueban las semillas "
            "[--seed, --seed + 1, ..., --seed + num_seeds - 1]."
        ),
    )

    parser.add_argument(
        "--msg-bits",
        type=int,
        default=0,
        choices=[0, 3],
        help="Bits de mensaje del entrenamiento: 0 (vanilla) o 3 (comms).",
    )

    parser.add_argument(
        "--time-limit",
        type=int,
        default=500,
        help="Número máximo de pasos del episodio.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS del video.",
    )

    return parser.parse_args()


LAYOUT = "x.x.x.x.x\n" * 8 + ".........\n..g...g.."


def build_env(time_limit, msg_bits):
    # small-4ag con el layout y parámetros del entrenamiento (custom)
    return gym.make(
        "rware-small-4ag-v2",
        layout=LAYOUT,
        msg_bits=msg_bits,
        sensor_range=1,
        request_queue_size=4,
        reward_type=RewardType.THREE_STAGE,
        render_mode="rgb_array",
        max_steps=time_limit,
    )


def run_episode(
    env,
    agent,
    seed,
    time_limit,
    n_agents,
    n_actions,
    obs_agent_id,
    obs_last_action,
    hidden_dim,
    render,
    action_nvec=None,
):
    """Ejecuta un episodio completo con una semilla dada.

    Si render=True, guarda los frames renderizados. Si render=False,
    se omite el renderizado para acelerar la búsqueda de la mejor semilla.

    Devuelve un diccionario con el resultado del episodio.
    """

    obs, info = env.reset(seed=seed)

    obs = np.asarray(obs, dtype=np.float32)

    if obs.ndim == 1:
        obs = obs.reshape(n_agents, -1)

    hidden_states = th.zeros(n_agents, hidden_dim)
    last_actions = th.zeros(n_agents, n_actions)

    frames = []

    total_reward = 0.0
    deliveries = 0
    collisions = 0
    step = -1

    if render:
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))

    for step in range(time_limit):

        obs_tensor = th.tensor(obs, dtype=th.float32)

        inputs = [obs_tensor]

        if obs_last_action:
            inputs.append(last_actions)

        if obs_agent_id:
            agent_ids = th.eye(n_agents)
            inputs.append(agent_ids)

        agent_inputs = th.cat(inputs, dim=1)

        with th.no_grad():
            q_values, hidden_states = agent(agent_inputs, hidden_states)

        actions = th.argmax(q_values, dim=1).cpu().numpy().astype(np.int64)

        if action_nvec is not None:
            # Índice plano -> [acción, bit1, bit2, bit3]
            env_actions = [
                np.array(np.unravel_index(a, action_nvec), dtype=np.int64)
                for a in actions
            ]
        else:
            env_actions = actions

        next_obs, reward, terminated, truncated, info = env.step(env_actions)

        if np.isscalar(reward):
            episode_reward = float(reward)
        else:
            episode_reward = float(np.sum(reward))

        total_reward += episode_reward

        if isinstance(info, dict):
            if "deliveries" in info:
                deliveries = info["deliveries"]
            if "collisions" in info:
                collisions = info["collisions"]

        obs = np.asarray(next_obs, dtype=np.float32)

        if obs.ndim == 1:
            obs = obs.reshape(n_agents, -1)

        if obs_last_action:
            last_actions.zero_()
            last_actions[np.arange(n_agents), actions] = 1.0

        if render:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))

        if terminated or truncated:
            break

    return {
        "seed": seed,
        "steps": step + 1,
        "total_reward": total_reward,
        "deliveries": deliveries,
        "collisions": collisions,
        "frames": frames,
    }


def main():
    args = parse_args()

    # ---------------------------------------------------------------
    # Configuración correspondiente al entrenamiento MAPPO (custom)
    # ---------------------------------------------------------------

    n_agents = 4
    hidden_dim = 128
    use_rnn = True

    obs_agent_id = True
    obs_last_action = True

    env = build_env(args.time_limit, args.msg_bits)

    # ---------------------------------------------------------------
    # Inicialización (para conocer tamaños de obs / acciones)
    # ---------------------------------------------------------------

    obs, info = env.reset(seed=args.seed)
    obs = np.asarray(obs, dtype=np.float32)

    if obs.ndim == 1:
        obs = obs.reshape(n_agents, -1)

    obs_size = obs.shape[-1]
    n_actions = get_agent_action_count(env.action_space)

    action_nvec = None
    if args.msg_bits > 0:
        action_nvec = tuple(int(n) for n in env.action_space[0].nvec)

    input_shape = obs_size

    if obs_last_action:
        input_shape += n_actions

    if obs_agent_id:
        input_shape += n_agents

    print(f"Observación por agente: {obs_size}")
    print(f"Input de RNNAgent: {input_shape}")
    print(f"Número de agentes: {n_agents}")
    print(f"Número de acciones: {n_actions}")

    # ---------------------------------------------------------------
    # Construcción de la red exactamente como BasicMAC
    # ---------------------------------------------------------------

    class Args:
        pass

    model_args = Args()
    model_args.hidden_dim = hidden_dim
    model_args.use_rnn = use_rnn
    model_args.n_actions = n_actions

    agent = RNNAgent(input_shape, model_args)

    # ---------------------------------------------------------------
    # Carga del checkpoint
    # ---------------------------------------------------------------

    model_path = os.path.join(args.checkpoint, "agent.th")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No se encontró agent.th en:\n{model_path}")

    state_dict = th.load(model_path, map_location="cpu")

    agent.load_state_dict(state_dict)
    agent.eval()

    print(f"\nModelo cargado: {model_path}")

    # ---------------------------------------------------------------
    # Búsqueda de la mejor semilla
    # ---------------------------------------------------------------

    num_seeds = max(1, args.num_seeds)
    if num_seeds > 1:
        candidate_seeds = list(range(args.seed, args.seed + num_seeds))
    else:
        candidate_seeds = [args.seed]

    print(
        f"\nEvaluando {num_seeds} semilla(s) "
        f"({candidate_seeds[0]}..{candidate_seeds[-1]}) para elegir la mejor..."
    )

    best_result = None

    for i, seed in enumerate(candidate_seeds):
        result = run_episode(
            env=env,
            agent=agent,
            seed=seed,
            time_limit=args.time_limit,
            n_agents=n_agents,
            n_actions=n_actions,
            obs_agent_id=obs_agent_id,
            obs_last_action=obs_last_action,
            hidden_dim=hidden_dim,
            # Solo renderizamos el episodio final elegido, no todos los
            # candidatos, para no gastar tiempo/memoria de más.
            render=False,
            action_nvec=action_nvec,
        )

        print(
            f"  [{i + 1}/{num_seeds}] semilla={seed:<6} "
            f"recompensa={result['total_reward']:.4f} "
            f"entregas={result['deliveries']} "
            f"colisiones={result['collisions']} "
            f"pasos={result['steps']}"
        )

        if best_result is None or result["total_reward"] > best_result["total_reward"]:
            best_result = result

    best_seed = best_result["seed"]

    print(
        f"\nMejor semilla: {best_seed} "
        f"(recompensa={best_result['total_reward']:.4f})"
    )
    print("Re-ejecutando el episodio con renderizado para guardar el video...")

    # ---------------------------------------------------------------
    # Re-ejecutar la mejor semilla, esta vez renderizando y guardando
    # ---------------------------------------------------------------

    final_result = run_episode(
        env=env,
        agent=agent,
        seed=best_seed,
        time_limit=args.time_limit,
        n_agents=n_agents,
        n_actions=n_actions,
        obs_agent_id=obs_agent_id,
        obs_last_action=obs_last_action,
        hidden_dim=hidden_dim,
        render=True,
        action_nvec=action_nvec,
    )

    env.close()

    frames = final_result["frames"]

    if not frames:
        raise RuntimeError(
            "El entorno no produjo frames. "
            "Comprueba que RWARE tenga render_mode='rgb_array'."
        )

    imageio.mimsave(args.output, frames, fps=args.fps)

    # ---------------------------------------------------------------
    # Resultados
    # ---------------------------------------------------------------

    print("\n" + "=" * 50)
    print("RESULTADO DEL EPISODIO (mejor semilla)")
    print("=" * 50)

    print(f"Semilla:     {final_result['seed']}")
    print(f"Pasos:       {final_result['steps']}")
    print(f"Recompensa:  {final_result['total_reward']:.4f}")
    print(f"Entregas:    {final_result['deliveries']}")
    print(f"Colisiones:  {final_result['collisions']}")
    print(f"Video:       {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()