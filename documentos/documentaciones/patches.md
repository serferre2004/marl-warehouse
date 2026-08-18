# Parches aplicados al código 

## EPyMARL

*2026-07-15*
### Archivo `src/envs/__init__.py`
- Se hicieron opcionales los imports de wrappers de entornos (SMACLite, etc.) mediante `try/except ImportError`, permitiendo ejecutar EPyMARL con solo las dependencias necesarias para RWARE.
### Archivo `src/main.py`
- Se añadió un `.replace(":", "_")` al string `f"sacred/{config_dict['name']}/{map_name}"` para evitar errores de guardado en Windows

*2026-08-10*
### Archivo `src/main.py`
- Se hicieron modificaciones para permitir una ruta personalizada para el almacenamiento de resultados. Se eliminó la línea:
    ``` python
    results_path = os.path.join(
        dirname(dirname(abspath(__file__))),
        "results"
    )
    ```
    y se reemplaza por:
    ``` python
    results_path = os.environ.get(
        "EPYMARL_RESULTS_PATH",
        os.path.join(dirname(dirname(abspath(__file__))), "results")
    )
    ```
### Archivo `src/run.py`
- Se añade comandos para sanitizar `unique_token` y evitar `OSError` por caracteres prohibidos en archivos
    ``` python
    unique_token = (
        f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}".replace(":", "_").replace(" ", "_")
    )
    ```

*2026-08-12*
### Archivo `config/default.yaml`
- Se agregan los parámetros de configuración de RWARE para permitir variaciones en los experimentos

    ``` yaml
    # --- RWARE environment parameters ---
    env_args:
    key: null
    time_limit: 500

    request_queue_size: null
    sensor_range: null
    msg_bits: null
    reward_type: null
    observation_type: null
    ```

### Archivo `envs/gymma.py`
- Se expande el espacio de acciones para ser compatible con los bits de comunicación (Expandir el espacio de acciones puede perjudicar el aprendizaje)

    ``` python
    # Método __init__
    # Reemplaza self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
    self.longest_action_space = max(self._env.action_space, key=self._action_space_size)

    # Nuevo método _action_space_size
    def _action_space_size(self, space):
        if hasattr(space, "n"):
            return space.n
        elif hasattr(space, "nvec"):
            return int(np.prod(space.nvec))
        else:
            raise TypeError(
                f"Unsupported action space type: {type(space)}"
            )

    # Método step
    # Reemplaza actions = [int(a) for a in actions]
    actions = [self._decode_action(agent_id, int(action)) for agent_id, action in enumerate(actions)]

    # Nuevo método _decode_action
    def _decode_action(self, agent_id, action):
        space = self._env.action_space[agent_id]

        if hasattr(space, "n"):
            return action

        if hasattr(space, "nvec"):
            nvec = space.nvec
            total_actions = int(np.prod(nvec))

            if action < 0 or action >= total_actions:
                raise ValueError(
                    f"Invalid action {action} for agent {agent_id}. "
                    f"Expected range [0, {total_actions - 1}]"
                )

            return np.array(
                np.unravel_index(action, tuple(nvec)),
                dtype=np.int64,
            )

        raise TypeError(
            f"Unsupported action space type: {type(space)}"
        )

    # Método get_avail_agent_actions
    def get_avail_agent_actions(self, agent_id):
        valid = self._action_space_size(self._env.action_space[agent_id])
        total = self._action_space_size(self.longest_action_space)
        invalid = total - valid
        return [1] * valid + [0] * invalid

    # Método get_total_actions
    def get_total_actions(self):
        """Returns the total number of discrete actions."""
        return self._action_space_size(self.longest_action_space)
    ```

## RWARE

*2026-07-18*
### Archivo `rware/warehouse.py`
- En la clase `Warehouse` se agrega el atributo `self.episode_collisions` para el conteo de colisiones. Se inicializa en 0 y se debe añadir la línea `self.episode_collisions = 0` en el método `reset`. Además se cuentan de la siguiente manera:
    ``` python
    ### Método step (línea 877)

            for agent in failed_agents:
                assert agent.req_action == Action.FORWARD
                agent.req_action = Action.NOOP
                self.episode_collisions += 1
    ```
    Para reportar el valor hacia el wrapper se modifica el método `_get_info`:
    ```python
    def _get_info(self):
        return {"episode_collisions": self.episode_collisions}
    ```

*2026-08-12*
### Archivo `rware/warehouse.py`
- Hay una inconsistencia en las observaciones cuando el parámetro msg_bits es 0. Porque no se construye el `observation_space` con la clave `msg_bits` pero el `default_obs` la reporta como None
    ``` python
    # Método _get_default_obs
    # Reemplaza obs["sensors"][i]["local_message"] = (
    #                self.msg_bits * [0] if self.msg_bits > 0 else None
    #            )
    if self.msg_bits > 0:
        obs["sensors"][i]["local_message"] = self.msg_bits * [0]

    # Reemplaza obs["sensors"][i]["local_message"] = (
    #                self.agents[id_ - 1].message if self.msg_bits > 0 else None
    #            )
    if self.msg_bits > 0:
        obs["sensors"][i]["local_message"] = self.agents[id_ - 1].message
    ```


*2026-08-17*
### Archivo `rware/warehouse.py`
- Para acelerar el aprendizaje se transforma el TWO_STAGE reward type por THREE_STAGE reward type en el que la recompensa a cada agente se entrega al recoger la estantería solicitada, entregarla y al dejarla en un espacio libre. Para evitar *reward hacking* se penaliza soltar una estantería solicitada sin haberla entregado.
    ``` python
    # Enumeración RewardType
    class RewardType(Enum):
        GLOBAL = 0
        INDIVIDUAL = 1
        THREE_STAGE = 2

    # Método step
    elif agent.req_action == Action.TOGGLE_LOAD and not agent.carrying_shelf:
        shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
        if shelf_id:
            agent.carrying_shelf = self.shelfs[shelf_id - 1]
            # Three stage reward for carrying a requested shelf
            if self.shelfs[shelf_id - 1] in self.request_queue and self.reward_type == RewardType.THREE_STAGE:
                rewards[agent.id - 1] += 0.3
    elif agent.req_action == Action.TOGGLE_LOAD and agent.carrying_shelf:
        if not self._is_highway(agent.x, agent.y):
            agent.carrying_shelf = None
            # We cancel the first stage reward to avoid reward hacking (self.grid x and y coordinates are inverted)
            shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
            if self.shelfs[shelf_id - 1] in self.request_queue and self.reward_type == RewardType.THREE_STAGE:
                rewards[agent.id - 1] -= 0.3
            if agent.has_delivered and self.reward_type == RewardType.THREE_STAGE:
                rewards[agent.id - 1] += 0.3

            agent.has_delivered = False

    # Método step
    shelf_delivered = True
    # remove from queue and replace it
    candidates = [s for s in self.shelfs if s not in self.request_queue]
    new_request = self.np_random.choice(candidates)
    self.request_queue[self.request_queue.index(shelf)] = new_request
    # also reward the agents
    if self.reward_type == RewardType.GLOBAL:
        rewards += 1
    elif self.reward_type == RewardType.INDIVIDUAL:
        agent_id = self.grid[_LAYER_AGENTS, x, y]
        rewards[agent_id - 1] += 1
    elif self.reward_type == RewardType.THREE_STAGE:
        agent_id = self.grid[_LAYER_AGENTS, x, y]
        self.agents[agent_id - 1].has_delivered = True
        rewards[agent_id - 1] += 0.4
    ```

- Para que se pueda pasar el reward_type como parámetro del comando se añade una transformación a la enumeración en Warehouse.__init__
    ``` python
    if type(reward_type) == str:
        reward_type = RewardType[reward_type]
    self.reward_type = reward_type
    ```