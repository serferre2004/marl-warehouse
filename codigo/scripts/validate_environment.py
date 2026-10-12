import argparse
from pathlib import Path

import gymnasium as gym
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Parámetros usados para construir la variante de RWARE.
# NO se pasan como kwargs a gym.make().
KEY_PARAMETERS = {
    "scenario",
    "n_agents",
}


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_rware_key(environment):
    base_key = environment["key"]
    scenario = environment["scenario"]
    n_agents = environment["n_agents"]

    # La key que utiliza RWARE/EPyMARL tiene esta forma:
    # rware:rware-tiny-2ag-v2
    return f"{base_key}-{scenario}-{n_agents}ag-v2"


def build_environment_kwargs(config):
    environment = config["environment"]

    return {
        key: value
        for key, value in environment.items()
        if key not in KEY_PARAMETERS and key != "key"
    }


def check_equal(name, expected, actual):
    if expected != actual:
        raise AssertionError(
            f"{name}: esperado={expected!r}, obtenido={actual!r}"
        )

    print(f"  ✓ {name}: {actual!r}")


def validate_environment(config):
    environment = config["environment"]

    key = build_rware_key(environment)
    kwargs = build_environment_kwargs(config)

    print("\n" + "=" * 65)
    print("VALIDACIÓN DE CONFIGURACIÓN RWARE")
    print("=" * 65)

    print("\n[1] Configuración de la variante")
    print(f"  Base key:  {environment['key']}")
    print(f"  Scenario:  {environment['scenario']}")
    print(f"  Agents:    {environment['n_agents']}")
    print(f"  Key final: {key}")

    print("\n[2] Kwargs que serán enviados a gym.make()")

    for name, value in kwargs.items():
        print(f"  {name}: {value!r}")

    print("\nCreando entorno...")

    env = gym.make(key, **kwargs)

    try:
        base_env = env.unwrapped

        print("\n[3] Parámetros estructurales")

        check_equal(
            "n_agents",
            environment["n_agents"],
            base_env.n_agents,
        )

        print("\n[4] Parámetros de RWARE")

        for parameter, expected in kwargs.items():
            if not hasattr(base_env, parameter):
                print(
                    f"  ! {parameter}: no se encontró como "
                    f"atributo directo en {type(base_env).__name__}"
                )
                continue

            actual = getattr(base_env, parameter)

            check_equal(
                parameter,
                expected,
                actual,
            )

        print("\n[5] Espacio de acciones")

        print(f"  {env.action_space}")

        print("\n[6] Espacio de observaciones")

        print(f"  {env.observation_space}")

        print("\n[7] Reset")

        observations, info = env.reset(
            seed=config["seed"]
        )

        print("  ✓ reset() ejecutado correctamente")
        print(f"  Tipo de observación: {type(observations)}")

        print("\n[8] Step")

        actions = env.action_space.sample()

        observations, reward, terminated, truncated, info = env.step(
            actions
        )

        print("  ✓ step() ejecutado correctamente")
        print(f"  Acción: {actions}")
        print(f"  Reward: {reward!r}")
        print(f"  Terminated: {terminated!r}")
        print(f"  Truncated: {truncated!r}")

        print("\n[9] Comunicación")

        msg_bits = environment.get("msg_bits")

        if msg_bits is not None:
            check_equal(
                "msg_bits",
                msg_bits,
                base_env.msg_bits,
            )

            if msg_bits == 0:
                print("  ✓ Comunicación desactivada")

            else:
                print(
                    f"  ✓ Comunicación activada "
                    f"({msg_bits} bits)"
                )

        print("\n[10] Recompensa")

        reward_config = config.get("reward", {})

        common_reward = reward_config.get("common")

        if common_reward is not None:
            print(
                f"  Configuración EPyMARL common_reward: "
                f"{common_reward}"
            )

        scalarisation = reward_config.get("scalarisation")

        if scalarisation is not None:
            print(
                f"  Configuración EPyMARL scalarisation: "
                f"{scalarisation}"
            )

        print("\n" + "=" * 65)
        print("ENVIRONMENT VALIDATION PASSED")
        print("=" * 65)

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Valida la configuración RWARE definida en un YAML."
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
            f"No se encontró el archivo: {config_path}"
        )

    config = load_config(config_path)

    print(f"\nConfiguración: {config_path}")

    validate_environment(config)


if __name__ == "__main__":
    main()