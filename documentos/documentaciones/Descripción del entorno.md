# Descripción del entorno

Creado por: Sergio
Fecha de creación: July 9, 2026 6:20 PM
Categoría: Documentación
Última edición por: Sergio
Fecha de última actualización: July 9, 2026 6:21 PM

---

## Configuraciones Disponibles

RWARE es parametrizable a través de la clase `Warehouse` o mediante los entornos pre-registrados en Gymnasium. Los parámetros principales son: [1](https://app.notion.com/p/Descripci-n-del-entorno-39827e02ef6f80c9a4e6f427be3fcf3c?pvs=21)

| Parámetro | Descripción | Valores típicos |
| --- | --- | --- |
| `shelf_columns` | Columnas de estanterías (debe ser impar) | 1, 3, 5… |
| `column_height` | Altura de cada columna de estanterías | 3, 8… |
| `shelf_rows` | Filas de bloques de estanterías | 1, 3… |
| `n_agents` | Número de robots | 2, 4, 6, 8 |
| `request_queue_size` | Estanterías solicitadas simultáneamente (R) | N, 2N, N/2 |
| `sensor_range` | Radio de visión del agente (celdas) | 1 (por defecto) |
| `msg_bits` | Bits de comunicación entre agentes | 0 (por defecto) |
| `max_steps` | Pasos máximos por episodio | None o entero |
| `max_inactivity_steps` | Pasos sin entrega antes de terminar | None o entero |
| `reward_type` | Tipo de recompensa | GLOBAL, INDIVIDUAL, TWO_STAGE |
| `observation_type` | Formato de observación | DICT, FLATTENED, IMAGE, IMAGE_DICT |
| `layout` | Layout personalizado como string | `"..x...\n.g...g."` |
| `normalised_coordinates` | Coordenadas normalizadas [0,1] | False (por defecto) |

### Entornos pre-registrados

Los nombres siguen el esquema `rware-<tamaño>-<N>ag[-<dificultad>]-v2`: [2](https://app.notion.com/p/Descripci-n-del-entorno-39827e02ef6f80c9a4e6f427be3fcf3c?pvs=21)

| Nombre | Tamaño de cuadrícula | Agentes (N) | Pedidos (R) |
| --- | --- | --- | --- |
| `rware-tiny-2ag-v2` | 10 × 11 | 2 | 2 (= N) |
| `rware-small-4ag-v2` | 10 × 20 | 4 | 4 (= N) |
| `rware-medium-6ag-v2` | 16 × 20 | 6 | 6 (= N) |
| `rware-large-8ag-v2` | 16 × 29 | 8 | 8 (= N) |
| `rware-tiny-2ag-easy-v2` | 10 × 11 | 2 | 4 (= 2N) |
| `rware-small-4ag-hard-v2` | 10 × 20 | 4 | 2 (= N/2) |

El tamaño de la cuadrícula se calcula como:

- **Alto**: `(column_height + 1) × shelf_rows + 2`
- **Ancho**: `(2 + 1) × shelf_columns + 1` [3](https://app.notion.com/p/Descripci-n-del-entorno-39827e02ef6f80c9a4e6f427be3fcf3c?pvs=21)

---

## Espacio de Observación

La observación es **parcialmente observable**: cada agente percibe únicamente un cuadrado de `(1 + 2 × sensor_range)²` celdas centrado en su posición. Con `sensor_range=1` (por defecto), esto equivale a una ventana de **3×3 = 9 celdas**. [4](https://app.notion.com/p/Descripci-n-del-entorno-39827e02ef6f80c9a4e6f427be3fcf3c?pvs=21)

### Información propia del agente (`self`) [5](https://app.notion.com/p/Descripci-n-del-entorno-39827e02ef6f80c9a4e6f427be3fcf3c?pvs=21)

| Campo | Tipo | Descripción |
| --- | --- | --- |
| `location` | `Box(2,)` | Coordenadas (x, y) del agente |
| `carrying_shelf` | `MultiBinary(1)` | 1 si lleva una estantería, 0 si no |
| `direction` | `Discrete(4)` | Orientación: UP=0, DOWN=1, LEFT=2, RIGHT=3 |
| `on_highway` | `MultiBinary(1)` | 1 si está en un pasillo/corredor |

### Información sensorial por celda (`sensors`)

Por cada una de las `(1 + 2×sensor_range)²` celdas del área de visión: [6](https://app.notion.com/p/Descripci-n-del-entorno-39827e02ef6f80c9a4e6f427be3fcf3c?pvs=21)

| Campo | Tipo | Descripción |
| --- | --- | --- |
| `has_agent` | `MultiBinary(1)` | 1 si hay otro agente en esa celda |
| `direction` | `Discrete(4)` | Dirección del agente en esa celda |
| `local_message` | `MultiBinary(msg_bits)` | Mensaje del agente (solo si `msg_bits > 0`) |
| `has_shelf` | `MultiBinary(1)` | 1 si hay una estantería en esa celda |
| `shelf_requested` | `MultiBinary(1)` | 1 si esa estantería está en la cola de pedidos |

### Tipos de observación disponibles [7](https://app.notion.com/p/Descripci-n-del-entorno-39827e02ef6f80c9a4e6f427be3fcf3c?pvs=21)

| Tipo | Descripción |
| --- | --- |
| `DICT` | Diccionario estructurado con claves `self` y `sensors` |
| `FLATTENED` | Vector 1D aplanado del diccionario anterior |
| `IMAGE` | Imagen multicanal de `(1+2×sensor_range) × (1+2×sensor_range)` |
| `IMAGE_DICT` | Imagen + vector de características adicionales |

Para el tipo `IMAGE`, las capas disponibles son: [8](https://app.notion.com/p/Descripci-n-del-entorno-39827e02ef6f80c9a4e6f427be3fcf3c?pvs=21)

| Capa | Descripción |
| --- | --- |
| `SHELVES` | Celdas con estanterías (binario) |
| `REQUESTS` | Estanterías solicitadas (binario) |
| `AGENTS` | Posiciones de agentes (binario) |
| `AGENT_DIRECTION` | Dirección de cada agente (entero) |
| `AGENT_LOAD` | Agentes que llevan carga (binario) |
| `GOALS` | Ubicaciones de entrega (binario) |
| `ACCESSIBLE` | Celdas accesibles, no ocupadas (binario) |

---

## Espacio de Acciones

Cada agente dispone de **5 acciones discretas**: [9](https://app.notion.com/p/Descripci-n-del-entorno-39827e02ef6f80c9a4e6f427be3fcf3c?pvs=21)

| Acción | Valor | Descripción |
| --- | --- | --- |
| `NOOP` | 0 | No hacer nada |
| `FORWARD` | 1 | Avanzar en la dirección actual |
| `LEFT` | 2 | Girar 90° a la izquierda (sentido antihorario) |
| `RIGHT` | 3 | Girar 90° a la derecha (sentido horario) |
| `TOGGLE_LOAD` | 4 | Cargar o descargar una estantería |

El espacio de acción global es `Tuple(Discrete(5), ..., Discrete(5))` con un elemento por agente. Si `msg_bits > 0`, cada acción individual pasa a ser `MultiDiscrete([5, 2, 2, ..., 2])` donde los bits adicionales son el mensaje a emitir. [10](https://app.notion.com/p/Descripci-n-del-entorno-39827e02ef6f80c9a4e6f427be3fcf3c?pvs=21)

**Restricciones de movimiento relevantes:**

- Un agente que lleva una estantería **solo puede moverse por pasillos** (highways).
- `TOGGLE_LOAD` para cargar solo funciona si hay una estantería en la celda actual.
- `TOGGLE_LOAD` para descargar solo funciona si el agente **no está en un pasillo**. [11](https://app.notion.com/p/Descripci-n-del-entorno-39827e02ef6f80c9a4e6f427be3fcf3c?pvs=21)

---

## Función de Recompensa

La recompensa es **escasa**: solo se emite cuando un agente entrega una estantería solicitada en una ubicación de meta (`G`). En cada paso sin entrega, la recompensa es 0. [12](https://app.notion.com/p/Descripci-n-del-entorno-39827e02ef6f80c9a4e6f427be3fcf3c?pvs=21)

RWARE soporta tres modalidades:

| Tipo | Quién recibe | Cuándo | Valor |
| --- | --- | --- | --- |
| `GLOBAL` | Todos los agentes | Al entregar cualquier estantería solicitada | +1.0 |
| `INDIVIDUAL` | Solo el agente que entrega | Al entregar una estantería solicitada | +1.0 |
| `TWO_STAGE` | Solo el agente que actúa | Al entregar en goal (+0.5) y al devolver la estantería a una celda no-pasillo (+0.5) | +0.5 + 0.5 |

Tras cada entrega exitosa, la estantería entregada se retira de la cola y se añade una nueva aleatoriamente, manteniendo siempre R pedidos activos. [13](https://app.notion.com/p/Descripci-n-del-entorno-39827e02ef6f80c9a4e6f427be3fcf3c?pvs=21)

**Esparsidad:** Con R pequeño y almacén grande, la probabilidad de que un agente aleatorio lleve la estantería correcta al lugar correcto es muy baja, lo que hace la exploración extremadamente difícil.
---

## Dinámica de un paso

Cada vez que la clase `Warehouse` ejecuta el método `step` el código se encarga de calcular la acción y mensaje para los demás agentes, luego por cada uno de los agentes se calculan las colisiones y se decide si permitir la acción o cancelarla incluyendo colisiones que busquen intercambiar la posición de dos agentes, luego se realiza el cambio pertinente para cada agente, finalmente se verifica que se haya entregado una estantería.


```
Calcula acción 
↓
Calcula colisiones con estanterías
↓
Calcula colisiones con agentes
↓
Ejecutar cambios
↓
Verificación de pedido entregado

```
---

## Episodios con Acciones Aleatorias

El siguiente código ejecuta varios episodios con política aleatoria y registra métricas básicas:

```python
import gymnasium as gym
import rware
import numpy as np

env = gym.make("rware-tiny-2ag-v2")
N_EPISODES = 20

all_rewards = []
all_lengths = []
all_deliveries = []

for ep in range(N_EPISODES):
    obs, info = env.reset()
    done = False
    ep_reward = np.zeros(env.unwrapped.n_agents)
    ep_steps = 0
    ep_deliveries = 0

    while not done:
        actions = env.action_space.sample()
        obs, rewards, done, truncated, info = env.step(actions)
        ep_reward += np.array(rewards)
        ep_steps += 1
        ep_deliveries += int(sum(rewards) > 0)  # pasos con al menos una entrega
        done = done or truncated

    all_rewards.append(ep_reward.sum())
    all_lengths.append(ep_steps)
    all_deliveries.append(ep_deliveries)
    print(f"Ep {ep+1:02d} | Pasos: {ep_steps:4d} | Recompensa total: {ep_reward.sum():.1f} | Entregas: {ep_deliveries}")

print("\n--- Métricas base (política aleatoria) ---")
print(f"Recompensa promedio por episodio : {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
print(f"Duración promedio (pasos)        : {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
print(f"Entregas promedio por episodio   : {np.mean(all_deliveries):.2f} ± {np.std(all_deliveries):.2f}")
print(f"Tasa de éxito (ep con ≥1 entrega): {np.mean([d > 0 for d in all_deliveries]):.2%}")

env.close()
```

**Comportamientos esperados con política aleatoria:**

- Las entregas son extremadamente raras o nulas, especialmente en configuraciones `hard`.
- Los agentes se bloquean frecuentemente entre sí (colisiones resueltas como NOOP).
- La duración del episodio depende de `max_steps` o `max_inactivity_steps`. [14](https://app.notion.com/p/Descripci-n-del-entorno-39827e02ef6f80c9a4e6f427be3fcf3c?pvs=21)

---

## Métricas Base (Línea de Referencia)

Las siguientes métricas deben registrarse para establecer la línea base con política aleatoria:

| Métrica | Descripción | Cómo calcularla |
| --- | --- | --- |
| **Recompensa promedio por episodio** | Suma de recompensas de todos los agentes en un episodio, promediada sobre episodios | `mean(sum(rewards) per episode)` |
| **Recompensa por agente** | Recompensa media individual | `mean(rewards[i]) per episode` |
| **Número de entregas por episodio** | Pasos en que al menos un agente entregó | `count(steps where sum(rewards) > 0)` |
| **Duración del episodio** | Número de pasos hasta terminación | `steps per episode` |
| **Tasa de éxito** | Fracción de episodios con al menos una entrega | `episodes with deliveries / total episodes` |
| **Tasa de bloqueo** | Fracción de pasos donde algún agente no pudo moverse | Requiere instrumentación adicional |

---

## Ficha Técnica del Entorno

| Campo | Valor |
| --- | --- |
| **Nombre del entorno** | `rware-tiny-2ag-v2` (ejemplo de referencia) |
| **Tamaño de la cuadrícula** | 10 × 11 celdas |
| **Número de agentes** | 2 |
| **Número de estanterías** | Todas las celdas no-pasillo (≈ 2/3 de la cuadrícula) |
| **Pedidos simultáneos (R)** | 2 (= N, dificultad por defecto) |
| **Radio de visión** | 1 celda → ventana 3×3 = 9 celdas |
| **Tipo de observación** | `FLATTENED` (vector 1D) por defecto |
| **Dimensión de observación** | `4 + 4 + 1 + 9×(1+4+1+1) = 72` valores por agente (con `sensor_range=1`, sin `msg_bits`) |
| **Espacio de acciones** | `Discrete(5)` por agente |
| **Acciones** | NOOP, FORWARD, LEFT, RIGHT, TOGGLE_LOAD |
| **Tipo de recompensa** | GLOBAL (por defecto) |
| **Valor de recompensa** | +1.0 por entrega exitosa; 0 en caso contrario |
| **Recompensa escasa** | Sí — solo al completar entregas |
| **Criterio de terminación** | `max_steps` alcanzado O `max_inactivity_steps` pasos sin entrega |
| **Observabilidad** | Parcial — cada agente ve solo su vecindad local |
| **Comunicación** | Desactivada por defecto (`msg_bits=0`) |
| **Recompensa base (aleatoria)** | ≈ 0.0 entregas/episodio (extremadamente escasa) |
| **Interfaz** | Gymnasium (`gym.make`) |
| **API de paso** | `obs, rewards, done, truncated, info = env.step(actions)` |

### Citations

**File:** rware/warehouse.py (L31-36)

```python
class Action(Enum):
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    TOGGLE_LOAD = 4
```

**File:** rware/warehouse.py (L46-56)

```python
class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2

class ObservationType(Enum):
    DICT = 0
    FLATTENED = 1
    IMAGE = 2
    IMAGE_DICT = 3
```

**File:** rware/warehouse.py (L59-71)

```python
class ImageLayer(Enum):
    """
    Input layers of image-style observations
    """

    SHELVES = 0  # binary layer indicating shelves (also indicates carried shelves)
    REQUESTS = 1  # binary layer indicating requested shelves
    AGENTS = 2  # binary layer indicating agents in the environment (no way to distinguish agents)
    AGENT_DIRECTION = 3  # layer indicating agent directions as int (see Direction enum + 1 for values)
    AGENT_LOAD = 4  # binary layer indicating agents with load
    GOALS = 5  # binary layer indicating goal/ delivery locations
    ACCESSIBLE = 6  # binary layer indicating accessible cells (all but occupied cells/ out of map)
```

**File:** rware/warehouse.py (L146-169)

```python
    def __init__(
        self,
        shelf_columns: int,
        column_height: int,
        shelf_rows: int,
        n_agents: int,
        msg_bits: int,
        sensor_range: int,
        request_queue_size: int,
        max_inactivity_steps: Optional[int],
        max_steps: Optional[int],
        reward_type: RewardType,
        layout: Optional[str] = None,
        observation_type: ObservationType = ObservationType.FLATTENED,
        image_observation_layers: List[ImageLayer] = [
            ImageLayer.SHELVES,
            ImageLayer.REQUESTS,
            ImageLayer.AGENTS,
            ImageLayer.GOALS,
            ImageLayer.ACCESSIBLE,
        ],
        image_observation_directional: bool = True,
        normalised_coordinates: bool = False,
        render_mode: str = "human",
```

**File:** rware/warehouse.py (L196-199)

```python
            The final grid size will be
            height: (column_height + 1) * shelf_rows + 2
            width: (2 + 1) * shelf_columns + 1
```

**File:** rware/warehouse.py (L255-260)

```python
        sa_action_space = [len(Action), *msg_bits * (2,)]
        if len(sa_action_space) == 1:
            sa_action_space = gym.spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = gym.spaces.MultiDiscrete(sa_action_space)
        self.action_space = gym.spaces.Tuple(tuple(n_agents * [sa_action_space]))
```

**File:** rware/warehouse.py (L432-443)

```python
        self._obs_bits_for_self = 4 + len(Direction)
        self._obs_bits_per_agent = 1 + len(Direction) + self.msg_bits
        self._obs_bits_per_shelf = 2
        self._obs_bits_for_requests = 2

        self._obs_sensor_locations = (1 + 2 * self.sensor_range) ** 2

        self._obs_length = (
            self._obs_bits_for_self
            + self._obs_sensor_locations * self._obs_bits_per_agent
            + self._obs_sensor_locations * self._obs_bits_per_shelf
        )
```

**File:** rware/warehouse.py (L460-469)

```python
        self_observation_dict_space = gym.spaces.Dict(
            OrderedDict(
                {
                    "location": location_space,
                    "carrying_shelf": gym.spaces.MultiBinary(1),
                    "direction": gym.spaces.Discrete(4),
                    "on_highway": gym.spaces.MultiBinary(1),
                }
            )
        )
```

**File:** rware/warehouse.py (L470-485)

```python
        sensor_per_location_dict = OrderedDict(
            {
                "has_agent": gym.spaces.MultiBinary(1),
                "direction": gym.spaces.Discrete(4),
            }
        )
        if self.msg_bits > 0:
            sensor_per_location_dict["local_message"] = gym.spaces.MultiBinary(
                self.msg_bits
            )
        sensor_per_location_dict.update(
            {
                "has_shelf": gym.spaces.MultiBinary(1),
                "shelf_requested": gym.spaces.MultiBinary(1),
            }
        )
```

**File:** rware/warehouse.py (L889-899)

```python
            elif agent.req_action == Action.TOGGLE_LOAD and not agent.carrying_shelf:
                shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
                if shelf_id:
                    agent.carrying_shelf = self.shelfs[shelf_id - 1]
            elif agent.req_action == Action.TOGGLE_LOAD and agent.carrying_shelf:
                if not self._is_highway(agent.x, agent.y):
                    agent.carrying_shelf = None
                    if agent.has_delivered and self.reward_type == RewardType.TWO_STAGE:
                        rewards[agent.id - 1] += 0.5

                    agent.has_delivered = False
```

**File:** rware/warehouse.py (L903-927)

```python
        shelf_delivered = False
        for y, x in self.goals:
            shelf_id = self.grid[_LAYER_SHELFS, x, y]
            if not shelf_id:
                continue
            shelf = self.shelfs[shelf_id - 1]

            if shelf not in self.request_queue:
                continue
            # a shelf was successfully delived.
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
            elif self.reward_type == RewardType.TWO_STAGE:
                agent_id = self.grid[_LAYER_AGENTS, x, y]
                self.agents[agent_id - 1].has_delivered = True
                rewards[agent_id - 1] += 0.5
```

**File:** rware/warehouse.py (L935-942)

```python
        if (
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        ) or (self.max_steps and self._cur_steps >= self.max_steps):
            done = True
        else:
            done = False
        truncated = False
```

**File:** [README.md](http://readme.md/) (L56-98)

```markdown
## Observation Space
The observation of an agent is partially observable and consists of a 3x3 (configurable) square centred on the agent. Inside this limited grid, all entities are observable:
- The location, the rotation and whether the agent is carrying a shelf.
- The location and rotation of other robots.
- Shelves and whether they are currently in the request queue.

## Dynamics: Collisions
The dynamics of the environment are also of particular interest. Like a real, 3-dimensional warehouse, the robots can move beneath the shelves. Of course, when the robots are loaded, they must use the corridors, avoiding any standing shelves.

Any collisions are resolved in a way that allows for maximum mobility. When two or more agents attempt to move to the same location, we prioritise the one that also blocks others. Otherwise, the selection is done arbitrarily. The visuals below demonstrate the resolution of various collisions.

 Example 1                 |   Example 2               | Example 3
:-------------------------:|:-------------------------:|:-------------------------:
![](docs/img/collision1.gif)  |  ![](docs/img/collision2.gif)  |  ![](docs/img/collision3.gif)

## Rewards
At each time a set number of shelves R is requested. When a requested shelf is brought to a goal location, another shelf is uniformly sampled and added to the current requests. Agents are rewarded for successfully delivering a requested shelf to a goal location, with a reward of 1. A significant challenge in these environments is for agents to deliver requested shelves but also finding an empty location to return the previously delivered shelf. Having multiple steps between deliveries leads a very sparse reward signal.

# Environment Parameters

The multi-robot warehouse task is parameterised by:

- The size of the warehouse which is preset to either tiny (10x11), small (10x20), medium (16x20), or large (16x29).
- The number of agents N.
- The number of requested shelves R. By default R=N, but easy and hard variations of the environment use R = 2N and R = N/2, respectively.

Note that R directly affects the difficulty of the environment. A small R, especially on a larger grid, dramatically affects the sparsity of the reward and thus exploration: randomly bringing the correct shelf becomes increasingly improbable.

## Naming Scheme

While RWARE allows fine tuning of multiple parameters when using the Warehouse class, it also registers multiple default environments with Gymnasium for simplicity.

The registered names look like `rware-tiny-2ag-v1` and might cryptic in the beginning, but it is not actually complicated. Every name always starts with rware. Next, the map size is appended as -tiny, -small, -medium, or -large. The number of robots in the map is selected as Xag with X being a number larger than one (e.g. -4ag for 4 agents). A difficulty modifier is optionally appended in the form of -easy or -hard, making requested shelves twice or half the number of agents (see section Rewards). Finally -v2 is the version as required for Gymnasium. In the time of writing all environments are v1, but we will increase it during changes or bugfixes.

A few examples:
```python
import gymnasium as gym
import rware

env = gym.make("rware-tiny-2ag-v2")
env = gym.make("rware-small-4ag-v2")
env = gym.make("rware-medium-6ag-hard-v2")
```

```

```