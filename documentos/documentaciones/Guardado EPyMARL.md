# Documentación de resultados del experimento EPyMARL (IQL + RWARE)

## Estructura de la carpeta `results`

Al ejecutar el experimento con `--config=iql --env-config=gymma env_args.key="rware:rware-tiny-2ag-v2"`, EPyMARL crea una carpeta de resultados en:  
```
epymarl/results/sacred/iql/rware_rware-tiny-2ag-v2/
```  
Dentro de esa ruta encontramos:  

- Una subcarpeta `_sources/` que contiene copias del código fuente usado (por ejemplo los scripts `main.py`, `run.py` y el módulo de logging). Estos archivos permiten reproducir exactamente el experimento.  
- Una subcarpeta numerada, en este caso `1/`, que corresponde al primer *run* del experimento. (Si hubieran múltiples ejecuciones independientes, habría varias carpetas numeradas).  

La carpeta `1/` incluye los archivos principales con los datos del run:

- `config.json`: **Configuración del experimento**. Contiene todos los hiperparámetros y valores usados (algoritmo, parámetros del entorno, seed, etc).  
- `cout.txt`: **Salida de consola completa**. Registra todo lo que se imprimió en pantalla durante el entrenamiento (logging informativo).  
- `metrics.json`: **Métricas de entrenamiento**. Contiene los valores registrados de cada métrica (por ejemplo rewards, pérdidas, etc) a lo largo de los pasos. Cada entrada tiene arrays de `steps` y `values` para cada métrica.  
- `info.json`: **Información extendida (opcional)**. Suele incluir datos adicionales del run o resultados de evaluación (por ejemplo, estadísticas detalladas). En algunos casos puede estar vacío o no existir si no se han producido entradas especiales.  
- `run.json`: **Metadatos de ejecución**. Aquí se guarda información del entorno de ejecución (fechas, tiempos totales, host, versión del código, IDs de Sacred, etc.), es útil para registro y reproducibilidad.  

Estas salidas se generan por defecto al usar Sacred para el logging interno de EPyMARL. En resumen, `config.json` registra la configuración completa, `cout.txt` es el log de consola, `metrics.json` almacena los valores de las métricas, y `run.json` guarda el estado y metadatos del run. (Por defecto, **también** se incluye el archivo `config.json` de Sacred y otros archivos menores, pero los anteriores son los más relevantes.)

## Checkpoints y modelos guardados

Por defecto, EPyMARL **no guarda modelos intermedios** a menos que se active la opción. Si se desea almacenar los checkpoints del entrenamiento, hay que configurar en el YAML:  
```yaml
save_model: True
save_model_interval:  <pasos de separación entre guardados>
```  
Cuando `save_model=True`, EPyMARL crea una carpeta `models/` dentro del directorio del run. En ella aparecerán subdirectorios nombrados por el número de timesteps transcurridos, cada uno conteniendo los archivos del modelo guardado hasta ese punto. 

Por ejemplo, podrías ver algo como:  
```
epymarl/results/sacred/iql/rware_rware-tiny-2ag-v2/1/models/100000/  (modelos tras 100k pasos)
epymarl/results/sacred/iql/rware_rware-tiny-2ag-v2/1/models/200000/
...
```  
Así, cada subcarpeta numérica es un *checkpoint*. Si no aparece ninguna carpeta `models/`, significa que no se habilitó el guardado (`save_model=False` por defecto). (Para cargar modelos, EPyMARL usa los parámetros `checkpoint_path` y `load_step` apuntando a esta carpeta).

## Contenido y resumen de los archivos JSON y logs

- **`config.json`**: recoge el diccionario de configuración final del experimento, incluyendo algoritmo (IQL), entorno (RWARE con clave `rware-tiny-2ag-v2`), límites de tiempo (`t_max`), tamaño de batch, arquitectura de red, entre otros parámetros. Es el estado completo de la configuración tras procesar los YAML.
- **`metrics.json`**: es un JSON donde cada clave es una métrica registrada (por ejemplo `train_return_mean`, `test_return_mean`, etc.). Cada métrica incluye un arreglo de `steps` (pasos de entrenamiento) y de `values` (valor de la métrica en ese paso). Esto permite graficar la evolución de las métricas a lo largo del entrenamiento.
- **`cout.txt`**: contiene todo el texto impreso en la salida estándar (stdout) durante la ejecución, útil para seguir logs detallados de progreso (p. ej. avisos de evaluación periódica, resumen por episodio, etc.).
- **`info.json`**: (si existe) suele almacenar registros adicionales generados por el `EpisodeRunner` o `Learner` de EPyMARL. Podría incluir, por ejemplo, estadísticas específicas de episodios de test o cualquier información adicional recopilada. En algunos experimentos puede estar vacío.
- **`run.json`**: guarda metadatos del run, como el identificador interno, fecha de inicio/fin, tiempo empleado, seed empleada, nombre del host y versión del código. Ayuda a reproducir o rastrear las condiciones de ejecución.  

> **Nota:** Según la documentación oficial de EPyMARL, los logs predeterminados incluyen *archivos de configuración, un JSON con todas las métricas, un TXT con los outputs y más*, tal como reflejamos arriba.

## Métricas: JSON, TensorBoard o CSV

Por defecto, EPyMARL escribe las métricas en **formato JSON** dentro de `metrics.json`. Sin embargo, el comportamiento se puede modificar en el archivo de configuración YAML:

- Para habilitar TensorBoard, se fija `use_tensorboard: True` en el config. De ese modo EPyMARL también generará logs compatibles con TensorBoard (por ejemplo, un archivo `events.out.tfevents...` dentro del run). Esto permite visualizar las métricas con la interfaz de TensorBoard.  
- El uso de CSV no está predeterminado en EPyMARL, pero se podría implementar exportando manualmente los datos o usando herramientas externas. (La documentación oficial no menciona directamente CSV, pero apunta a TensorBoard y W&B como alternativas).  

En resumen: **JSON es el formato por defecto** para métricas, pero el usuario puede configurar `use_tensorboard: True` para exportar también a TensorBoard. Además, mediante herramientas externas (p. ej. pandas) se podrían convertir estos datos JSON a CSV si se requiere.

## Resumen de rutas importantes

- **Carpeta de resultados principal:** `epymarl/results/sacred/iql/rware_rware-tiny-2ag-v2/`.  
- **Código fuente del experimento:** en `_sources/`, contiene `main.py`, `run.py`, etc.  
- **Datos del run #1:** en `epymarl/results/sacred/iql/rware_rware-tiny-2ag-v2/1/`. Ahí están los archivos `config.json`, `cout.txt`, `info.json`, `metrics.json`, `run.json`.  
- **Modelos guardados (checkpoints):** si se habilita `save_model`, aparecerá `.../1/models/<pasos>/` con cada checkpoint.  
- **Formatos de métricas:** JSON por defecto (`metrics.json`), con opción a TensorBoard vía config.

Este documento puede guardarse como referencia (por ejemplo `README_resultados.md`) dentro del proyecto EPyMARL para futuras consultas. En él quedan documentadas las rutas y contenidos clave generados durante el experimento.  

**Referencias:** Según la documentación oficial de EPyMARL, así es como se registran los archivos y métricas. Además, prácticas de usuarios en foros y blogs confirman que `config.json`, `cout.txt`, `metrics.json` y `run.json` corresponden a configuración, salida, métricas y estado de ejecución.