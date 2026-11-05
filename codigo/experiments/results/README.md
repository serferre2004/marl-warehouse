# Resultados de entrenamiento

Cada escenario sigue la misma estructura:

    <escenario>/sacred/<algoritmo>/rware_rware-<variante>-<n>ag-v2/seed<N>/
    <escenario>/models/<algoritmo>_seed<N>_rware_rware-<variante>-<n>ag-v2_<fecha>/<paso>/

| Carpeta | Contenido |
|---|---|
| `tiny/` | tiny, 2 y 4 agentes |
| `small/` | small, 2 y 4 agentes, 40M de pasos |
| `custom/` | layout custom, 4 agentes, `vanilla` y `comms`, 40M de pasos |
| `archive/small_leftovers/` | restos anteriores de small; no se usan en el análisis |

Análisis y figuras: `analysis/marl_pipeline/data/<escenario>/` y `analysis/marl_pipeline/figures/<escenario>/current/`.
