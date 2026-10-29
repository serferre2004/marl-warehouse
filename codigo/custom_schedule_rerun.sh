log_file="custom_schedule_lin.log"
write_log() {
    local timestamp
    timestamp=$(date +"%m-%d %H:%M:%S")
    echo "$timestamp - $1" >> "$log_file"
}

for seed in 3 22 42 138 2026; do
    (
        config="mappo_custom_comms_seed${seed}.yaml"
        write_log "Running experiment with config: $config"
        python3 scripts/run_experiment.py experiments/configs/"$config"
        write_log "Finished experiment with config: $config"
    ) &
done
wait
