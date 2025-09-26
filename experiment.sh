# Save the config into log, including what model , dataset, and hyperparameters
# Run the experiment
source mskh_env/bin/activate

#timestamp 
timestamp=
mode = "decompose"

#Add output name for pipeline code
python3 src/pipeline.py --mode $mode