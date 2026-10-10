$LogFile = "custom_schedule.log"
function Write-Log {
    param([string]$Message)
    $TimeStamp = Get-Date -Format "MM-dd HH:mm:ss"
    "$TimeStamp - $Message" | Out-File -FilePath $LogFile -Append
}

$scheduled = @("ia2c_custom_seed138", "ippo_custom_seed3", "maa2c_custom_seed3", "mappo_custom_seed138")
foreach ($config in $scheduled) {
    Write-Host "Running experiment with config: $config"
    Write-Log "Running experiment with config: $config"
    $config = "$($config).yaml"
    python scripts/run_experiment.py experiments/configs/$config
    Write-Log "Finished experiment with config: $config"
}
