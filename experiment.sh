#!/bin/bash

# Save the config into log, including what model, dataset, and hyperparameters
# Run the experiment
source mskh_env/bin/activate

# Generate timestamp for unique log files
timestamp=$(date +"%Y%m%d_%H%M%S")
mode="decompose"

# Create logs directory if it doesn't exist
mkdir -p logs

# Define log file path
log_file="logs/experiment_${mode}_${timestamp}.log"
config_file="logs/config_${mode}_${timestamp}.json"

# Log experiment configuration
echo "Experiment Configuration - $(date)" > $log_file
echo "=================================" >> $log_file
echo "Mode: $mode" >> $log_file
echo "Timestamp: $timestamp" >> $log_file
echo "Python Version: $(python3 --version)" >> $log_file
echo "=================================" >> $log_file
echo "" >> $log_file

# Save detailed configuration to JSON file
cat > $config_file << EOF
{
    "experiment_id": "${mode}_${timestamp}",
    "timestamp": "$(date -Iseconds)",
    "mode": "$mode",
    "python_version": "$(python3 --version)"
}
EOF

echo "Starting experiment with mode: $mode"
echo "Log file: $log_file"
echo "Config file: $config_file"

# Run the experiment with logging
echo "Starting pipeline execution at $(date)" >> $log_file
python3 src/pipeline.py \
    --mode $mode \
    --experiment_id "${mode}_${timestamp}" \
    --dataset "gender_bias" \
    --corpus "polnli" \
    --k 5 \
    --num_questions 5 \
    --corpus_size 10 \
    2>&1 | tee -a $log_file

# Log completion
echo "" >> $log_file
echo "Experiment completed at $(date)" >> $log_file
echo "Exit code: ${PIPESTATUS[0]}" >> $log_file

echo "Experiment completed. Check $log_file for details."