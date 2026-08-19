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
        help="Semilla del episodio.",
    )

    parser.add_argument(
        "--time-limit",
        type=int,
        default=1000,
        help="Número máximo de pasos del episodio.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS del video.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ---------------------------------------------------------------
    # Configuración correspondiente al entrenamiento IA2C
    # ---------------------------------------------------------------

    n_agents = 2
    hidden_dim = 128
    use_rnn = True

    obs_agent_id = True
    obs_last_action = True

    # tiny-2ag
    env = gym.make(
        "rware-tiny-2ag-v2",
        reward_type=RewardType.THREE_STAGE,
        render_mode="rgb_array",
        max_steps=args.time_limit,
    )

    # ---------------------------------------------------------------
    # Inicialización
    # ---------------------------------------------------------------

    obs, info = env.reset(seed=args.seed)

    # Gymnasium puede devolver distintas estructuras dependiendo
    # de la versión de RWARE.
    obs = np.asarray(obs, dtype=np.float32)

    if obs.ndim == 1:
        obs = obs.reshape(n_agents, -1)

    obs_size = obs.shape[-1]
    n_actions = get_agent_action_count(env.action_space)

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
        raise FileNotFoundError(
            f"No se encontró agent.th en:\n{model_path}"
        )

    state_dict = th.load(
        model_path,
        map_location="cpu",
    )

    agent.load_state_dict(state_dict)
    agent.eval()

    print(f"\nModelo cargado: {model_path}")

    # ---------------------------------------------------------------
    # Estado oculto de cada agente
    # BasicMAC utiliza:
    #
    # [batch, n_agents, hidden_dim]
    # ---------------------------------------------------------------

    hidden_states = th.zeros(
        n_agents,
        hidden_dim,
    )

    # Últimas acciones, solamente necesarias si obs_last_action=True.
    last_actions = th.zeros(
        n_agents,
        n_actions,
    )

    frames = []

    total_reward = 0.0
    deliveries = 0
    collisions = 0

    # Primer frame
    frame = env.render()

    if frame is not None:
        frames.append(np.asarray(frame))

    # ---------------------------------------------------------------
    # Ejecutar episodio
    # ---------------------------------------------------------------

    for step in range(args.time_limit):

        obs_tensor = th.tensor(
            obs,
            dtype=th.float32,
        )

        inputs = []

        # Observación
        inputs.append(obs_tensor)

        # Última acción
        if obs_last_action:
            inputs.append(last_actions)

        # ID del agente
        if obs_agent_id:
            agent_ids = th.eye(n_agents)
            inputs.append(agent_ids)

        # Igual que BasicMAC._build_inputs():
        # concatenamos las entradas de cada agente.
        agent_inputs = th.cat(inputs, dim=1)

        # Forward del GRU.
        with th.no_grad():
            q_values, hidden_states = agent(
                agent_inputs,
                hidden_states,
            )

        # -----------------------------------------------------------
        # Política greedy
        # -----------------------------------------------------------

        actions = th.argmax(
            q_values,
            dim=1,
        ).cpu().numpy()

        actions = actions.astype(np.int64)

        # -----------------------------------------------------------
        # Step
        # -----------------------------------------------------------

        next_obs, reward, terminated, truncated, info = env.step(
            actions
        )

        # RWARE puede devolver reward global o vectorial dependiendo
        # de reward_type.
        if np.isscalar(reward):
            episode_reward = float(reward)
        else:
            episode_reward = float(np.sum(reward))

        total_reward += episode_reward

        # -----------------------------------------------------------
        # Información adicional
        # -----------------------------------------------------------

        if isinstance(info, dict):

            # Algunas versiones/implementaciones de RWARE exponen
            # información de entregas mediante info.
            if "deliveries" in info:
                deliveries = info["deliveries"]

            if "collisions" in info:
                collisions = info["collisions"]

        obs = np.asarray(
            next_obs,
            dtype=np.float32,
        )

        if obs.ndim == 1:
            obs = obs.reshape(n_agents, -1)

        if obs_last_action:
            last_actions.zero_()
            last_actions[
                np.arange(n_agents),
                actions,
            ] = 1.0

        # Frame
        frame = env.render()

        if frame is not None:
            frames.append(np.asarray(frame))

        if terminated or truncated:
            break

    env.close()

    # ---------------------------------------------------------------
    # Guardar video
    # ---------------------------------------------------------------

    if not frames:
        raise RuntimeError(
            "El entorno no produjo frames. "
            "Comprueba que RWARE tenga render_mode='rgb_array'."
        )

    imageio.mimsave(
        args.output,
        frames,
        fps=args.fps,
    )

    # ---------------------------------------------------------------
    # Resultados
    # ---------------------------------------------------------------

    print("\n" + "=" * 50)
    print("RESULTADO DEL EPISODIO")
    print("=" * 50)

    print(f"Pasos:       {step + 1}")
    print(f"Recompensa:  {total_reward:.4f}")
    print(f"Entregas:    {deliveries}")
    print(f"Colisiones:  {collisions}")
    print(f"Video:       {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
