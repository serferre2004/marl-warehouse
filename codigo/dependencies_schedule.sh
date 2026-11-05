#!/usr/bin/env bash
# Experimentos con dependencias: small, 4 agentes, 6 pedidos en cola, p = 0.9.
# Lanza todos los algoritmos y semillas, con 3 entrenamientos a la vez.
# Uso, desde codigo/:  bash dependencies_schedule.sh
log_file="dependencies_schedule_lin.log"
output_dir="dependencies_outputs"
max_parallel=3
mkdir -p "$output_dir"

write_log() {
    echo "$(date +"%m-%d %H:%M:%S") - $1" >> "$log_file"
}

run_config() {
    local config="$1"
    write_log "Running experiment with config: $config"
    if python3 scripts/run_experiment.py experiments/configs/"$config" > "$output_dir/${config%.yaml}.out" 2>&1; then
        write_log "Finished experiment with config: $config"
    else
        write_log "FAILED experiment with config: $config"
    fi
}

for algorithm in ia2c ippo maa2c mappo; do
    for seed in 3 22 42 138 2026; do
        # Si ya hay 3 en marcha, espera a que termine uno
        while [ "$(jobs -rp | wc -l)" -ge "$max_parallel" ]; do
            wait -n
        done
        run_config "${algorithm}_small_dep_p09_seed${seed}.yaml" &
    done
done
wait
write_log "All dependency experiments finished"
