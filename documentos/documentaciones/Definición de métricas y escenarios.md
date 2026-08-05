# Definición de métricas y escenarios

Creado por: Sergio
Fecha de creación: July 11, 2026 1:35 PM
Última edición por: Sergio
Fecha de última actualización: July 11, 2026 2:35 PM

## Métricas

| **Nombre** | **Importancia** | **Requiere modificación** | **Variable** | **Permite comparar** |
| --- | --- | --- | --- | --- |
| Recompensa promedio por episodio | Alta | No | Cumplimiento de la tarea en términos generales | Diferentes algoritmos o hiperparámetros para mismo escenario |
| Recompensa por agente | Alta | No | Eficiencia por agente | Escenarios con distinto número de agentes |
| Pasos promedio por entrega | Alta | No  | Eficiencia general | Diferentes algoritmos o hiperparámetros para mismo escenario |
| Recompensa durante entrenamiento | Alta | No | Capacidad de aprendizaje | Distintos algoritmos o hiperparámetros para mismo escenario |
| Varianza entre semillas | Media | No | Estabilidad de algoritmos | Diferentes algoritmos para mismo escenario |
| Número de colisiones | Media | Sí | Capacidad de coordinación | Distintos algoritmos para mismo escenario |

---

## Escenarios

### Escenarios estándar

| # | Tamaño | Número de agentes | Observación | Recompensa | Comunicación | Rango sensor | Pedidos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | tiny (10x11) | 2 | FLATTENED | GLOBAL | 0 | 1 | 2 |
| 2 | medium (16x20) | 6 | FLATTENED | GLOBAL | 0 | 1 | 6 |
| 3 | large (16x29) | 8 | FLATTENED | GLOBAL | 0 | 1 | 8 |
| 4 | large (16x29) | 8 | FLATTENED | GLOBAL | 0 | 1 | 14 |

### Escenarios de variaciones

| # | Tamaño | Número de agentes | Observación | Recompensa | Comunicación | Rango sensor | Pedidos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | medium (16x20) | 2 | FLATTENED | INDIVIDUAL | 0 | 1 | 2 |
| 2 | medium (16x20) | 3 | FLATTENED | INDIVIDUAL | 0 | 1 | 3 |
| 3 | medium (16x20) | 3 | FLATTENED | TWO-STAGE | 0 | 1 | 3 |
| 4 | medium (16x20) | 3 | FLATTENED | INDIVIDUAL | 0 | 1 | 5 |
| 5 | medium (16x20) | 3 | FLATTENED | INDIVIDUAL | 4 | 1 | 3 |