# Protocolo experimental
Resumen ejecutivo: Proponemos un protocolo detallado para la fase experimental en el entorno RWARE con los algoritmos IQL, IA2C, MAA2C y MAPPO (implementados en EPyMARL). Se establecen decisiones específicas sobre semillas, duración de entrenamiento, frecuencia de evaluación y puntos de control, siguiendo recomendaciones de la literatura y prácticas de EPyMARL. Se detallan los parámetros de entrenamiento, métricas, registro de resultados y ajustes de hiperparámetros. Además incluimos una plantilla de configuración YAML, una tabla de alternativas (impacto computacional vs rigor) y un checklist de buenas prácticas para garantizar la reproducibilidad. En resumen, todas las comparaciones se harán bajo condiciones equivalentes (mismos escenarios, semillas y parámetros) y reportando medias con intervalos de confianza y análisis estadístico apropiado.

## Algoritmos

- IQL
- IA2C
- MAA2C
- MAPPO

Todos los algoritmos tienen implementaciones en https://github.com/uoe-agents/epymarl

## Stack tecnológico

```
Python 3.11 + Librerias estándar (matplotlib, numpy, etc.)
Gymnasium
RWARE (Modificado)
PyTorch
EpyMARL
```

## Parámetros de entrenamiento

| **Parámetro** | **Valor** |
| --- | --- |
| Número de semillas | 5 |
| Pasos de entrenamiento  | 2e6 |
| Frecuencia de evaluación | 5e3 |
| Número de episodios | 50 |
| Protocolo de sintonización | Random search |

## Sintonización de hiperparámetros

Cada uno se registrará en una etapa posterior

> La sintonización se realizará individualmente para cada algoritmo utilizando únicamente los escenarios estándar. Una vez definidos los hiperparámetros finales, estos permanecerán fijos para todos los escenarios de validación.
> 

### IQL

### IA2C

### MAA2C

### MAPPO

## Protocolo de entrenamiento y evaluación

1. **Configuración inicial:** Por cada experimento (algoritmo × escenario × semilla): preparar el archivo de configuración YAML. Incluir claves de hardware, versiones, semilla, hyperparámetros, etc. EPyMARL aplica automáticamente la semilla global y para `env.reset()` usando `with seed=<n>`.
2. **Entrenamiento:** Correr el entrenamiento hasta alcanzar los pasos totales definidos. EPyMARL registra automáticamente métricas (configuración, recompensas, pérdidas) en formato JSON y logs de texto. Se deben guardar puntos de control periódicamente (`save_model=True` y `save_model_interval`, e.g. cada 50000 pasos). Esto crea subdirectorios con los pesos aprendidos en `results/.../models/`.
3. **Evaluación periódica:** Cada 10000 pasos, detener temporalmente el entrenamiento y evaluar la política actual durante 50 episodios. Calcular la recompensa promedio y otras métricas (p.ej. pasos por entrega) en esa evaluación. Registrar media y varianza (o IC) sobre los 50 episodios. Continuar entrenando tras la evaluación.
4. **Evaluación final:** Al terminar el entrenamiento, cargar el último o mejor checkpoint registrado y evaluarlo *más exhaustivamente*. Por ejemplo, ejecutar ~500 episodios (10× mayor que en evaluaciones intermedias) para obtener estimaciones finales muy estables. Reportar promedio y IC95% de esa evaluación final.
5. **Repetición por semillas:** Repetir los pasos 1–4 para cada una de las semillas (usando semilla distinta cada vez). Es importante usar los mismos valores de configuración en cada semilla, variando sólo la semilla RNG.
6. **Agregación de resultados:** Para cada combinación algoritmo+escenario, calcular la recompensa promedio total (promedio de medias entre semillas) y su IC95%. Se recomienda reportar la media con IC95% (bootstrap estratificado sobre semillas). Para comparar entre algoritmos o escenarios, puede usarse la Media Intercuartílica (IQM) con IC (mejor para múltiples tareas).
7. **Test estadístico:** Si se desea inferencia estadística, aplicar pruebas no paramétricas (p.ej. prueba de Wilcoxon) o bootstrap para comparar distribuciones de recompensa entre algoritmos. *No* basarse solo en medias puntuales; con pocas semillas (<10) es preferible intervalos de confianza y prob. de mejora en lugar de tests t clásicos.

*Ejemplo de comandos EPyMARL:*

```bash
*python3 src/main.py --config=iql --env-config=gymma \
    with seed=0 env_args.time_limit=500 env_args.key="rware:rware-tiny-2ag-v2" \
    save_model=True save_model_interval=50000 t_max=2000000 evaluate_interval=100000*
```

*Este comando entrena IQL en RWARE tiny 2ag durante 2M pasos, evalúa cada 100k, guarda modelo cada 50k, usando la semilla 0. (Para IA2C, MAA2C, MAPPO, cambiar --config=.)*

## Plantilla de configuración (YAML/JSON)

A continuación un ejemplo de configuración (formato YAML) que reúne los parámetros clave del experimento:

```yaml
experiment:
  name: MARL_RWARE_comparison
  description: |
    Entrenamiento de IQL en RWARE_small_4ag, semillas múltiples.  framework:
    python: "3.11"
    pytorch: "2.0"
    epymarl_commit: "abcd1234"
    rware_commit: "ef567890"
  algorithm:
    name: IQL
    hyperparams:
      gamma: 0.99
      learning_rate: 0.0005
      batch_size: 32
      optimizer: Adam
  environment:
    name: RWARE_small_4ag_v2
    observation: FLATTENED
    common_reward: true
    time_limit: 500
  training:
    total_timesteps: 2000000
    eval_interval: 10000
    eval_episodes: 50
    save_model: true
    save_interval: 50000
    seeds: [0, 1, 2, 3, 4]
    device: "cuda:0"
  logging:
    use_tensorboard: true
    use_wandb: false
```

Este YAML es ilustrativo. En EPyMARL se suelen usar varios archivos (`config=...` y `env-config=...`), pero aquí condensamos los parámetros críticos. Se debe incluir además: identificadores de commit, paquetes/librerías versionadas, semilla fija global (por reproducibilidad), y rutas de salida.

## Fichero de resultados

```
experiments/

    standard_1/

        IQL/
        
	        seed0/
	        seed1/
	        ...
        IA2C/
        MAA2C/
        MAPPO/

    standard_2/

    standard_3/

    standard_4/

results/

    csv/
    
	    standard1/
	    
		    IQL/
			    
			    seed0/
			    seed1/
			    ...

    figures/

    tables/

models/

logs/
```

## Figuras esperadas

Preeliminares, sujeto a cambios

- Recompensa durante entrenamiento.
- Recompensa final por algoritmo.
- Entregas por episodio.
- Pasos por entrega.
- Boxplot entre semillas.
- Comparación por escenario.
- Comparación por número de agentes.
- Comparación por dificultad.