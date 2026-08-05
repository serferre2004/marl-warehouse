# Parches aplicados al código 

## EPyMARL

*2026-07-15*
### Archivo `src/envs/__init__.py`
- Se hicieron opcionales los imports de wrappers de entornos (SMACLite, etc.) mediante `try/except ImportError`, permitiendo ejecutar EPyMARL con solo las dependencias necesarias para RWARE.
### Archivo `src/main.py`
- Se añadió un `.replace(":", "_")` al string `f"sacred/{config_dict['name']}/{map_name}"` para evitar errores de guardado en Windows


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