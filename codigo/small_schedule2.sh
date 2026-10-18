log_file="small_schedule_lin.log"
write_log() {
    local message="$1"
    local timestamp
    timestamp=$(date +"%m-%d %H:%M:%S")
    echo "$timestamp - $message" >> "$log_file"
}
algorithm="ippo"
seeds=("3" "22" "42" "138" "2026")
for seed in "${seeds[@]}"; do
    config="${algorithm}_small_seed${seed}.yaml"
    echo "Running experiment with config: $config"
    write_log "Running experiment with config: $config"
    python3 scripts/run_experiment.py experiments/configs/"$config"
    write_log "Finished experiment with config: $config"
done
