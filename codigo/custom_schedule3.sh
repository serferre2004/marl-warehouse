log_file="custom_schedule_lin.log"
write_log() {
    local message="$1"
    local timestamp
    timestamp=$(date +"%m-%d %H:%M:%S")
    echo "$timestamp - $message" >> "$log_file"
}

scheduled=("ippo_custom_vanilla_seed42" "ippo_custom_comms_seed42" "maa2c_custom_vanilla_seed42" "maa2c_custom_comms_seed42" "mappo_custom_vanilla_seed42" "mappo_custom_comms_seed42")
for config in "${scheduled[@]}"; do
    echo "Running experiment with config: $config"
    write_log "Running experiment with config: $config"
    config="$config.yaml"
    python3 scripts/run_experiment.py experiments/configs/"$config"
    write_log "Finished experiment with config: $config"
done
