#!/bin/bash

# Default values
DEVICES=(2) # Modify this array with your available GPU IDs
NUM_EXPERIMENTS=16
MAX_JOBS_PER_GPU=2  # Maximum number of concurrent jobs per GPU

# Configuration
DATASET="cifar10"
EPOCHS=100
SAVE_STEPS=20
ARCH="wrn28-2"
EXCLUDE_CLASS=0
LOGDIR="exp{:d}/cifar10".format(EXCLUDE_CLASS)
LOG_DIR="logs"

# Create logs directory if it doesn't exist
mkdir -p $LOG_DIR

echo "Starting $NUM_EXPERIMENTS training jobs across ${#DEVICES[@]} GPUs (${DEVICES[@]})"
echo "Maximum $MAX_JOBS_PER_GPU concurrent jobs per GPU"

# Array to track number of active jobs on each GPU
declare -A active_jobs
for gpu in "${DEVICES[@]}"; do
  active_jobs[$gpu]=0
done

# Array to store job PIDs by GPU
declare -A gpu_jobs
for gpu in "${DEVICES[@]}"; do
  gpu_jobs[$gpu]=""
done

# Get the next available GPU
get_available_gpu() {
  local least_busy_gpu=""
  local min_jobs=$MAX_JOBS_PER_GPU
  
  for gpu in "${DEVICES[@]}"; do
    # Count actual running jobs for this GPU
    local current_jobs=0
    
    if [[ -n "${gpu_jobs[$gpu]}" ]]; then
      for pid in ${gpu_jobs[$gpu]}; do
        if kill -0 $pid 2>/dev/null; then
          ((current_jobs++))
        fi
      done
    fi
    
    # Update the active job count with the actual number
    active_jobs[$gpu]=$current_jobs
    
    # Check if this GPU has capacity and fewer jobs than others
    if [[ $current_jobs -lt $min_jobs ]]; then
      min_jobs=$current_jobs
      least_busy_gpu=$gpu
    fi
  done
  
  # If found a GPU with capacity, return it
  if [[ $min_jobs -lt $MAX_JOBS_PER_GPU ]]; then
    echo $least_busy_gpu
    return 0
  fi
  
  # No GPU with capacity found
  return 1
}

# Main job scheduling loop
EXPID=0
while [[ $EXPID -lt $NUM_EXPERIMENTS ]]; do
  # Try to get an available GPU
  GPU_ID=$(get_available_gpu) || GPU_ID=""
  
  if [[ -n "$GPU_ID" ]]; then
    echo "Starting experiment $EXPID on GPU $GPU_ID ($(date))"
    
    # Run the training command
    CUDA_VISIBLE_DEVICES=$GPU_ID python3.10 -u train.py \
        --dataset=$DATASET \
        --epochs=$EPOCHS \
        --save_steps=$SAVE_STEPS \
        --arch $ARCH \
        --num_experiments $NUM_EXPERIMENTS \
        --expid $EXPID \
        --logdir $LOGDIR \
        --exclude_class $EXCLUDE_CLASS \
        &> ${LOG_DIR}/log_${EXPID} &
    
    # Get PID of background process
    job_pid=$!
    
    # Add PID to the GPU's job list
    gpu_jobs[$GPU_ID]="${gpu_jobs[$GPU_ID]} $job_pid"
    
    # Update active jobs count
    active_jobs[$GPU_ID]=$((active_jobs[$GPU_ID] + 1))
    
    echo "Experiment $EXPID started with PID $job_pid on GPU $GPU_ID (${active_jobs[$GPU_ID]}/${MAX_JOBS_PER_GPU} jobs)"
    
    # Move to next experiment
    EXPID=$((EXPID + 1))
    
    # Small delay to stagger job starts
    sleep 2
  else
    echo "All GPUs at capacity. Waiting for jobs to complete..."
    sleep 10
    
    # Display running jobs status
    echo "Current GPU usage:"
    for gpu in "${DEVICES[@]}"; do
      echo "  GPU $gpu: ${active_jobs[$gpu]}/${MAX_JOBS_PER_GPU} jobs"
    done
  fi
done

echo "All training jobs submitted! Monitoring progress..."

# Wait for all jobs to complete
all_done=false
while ! $all_done; do
    all_done=true
    jobs_running=0
    
    # Check each GPU's jobs
    for gpu in "${DEVICES[@]}"; do
        active=0
        if [[ -n "${gpu_jobs[$gpu]}" ]]; then
            for pid in ${gpu_jobs[$gpu]}; do
                if kill -0 $pid 2>/dev/null; then
                    ((active++))
                    ((jobs_running++))
                    all_done=false
                fi
            done
        fi
        echo "GPU $gpu: $active jobs running"
    done
    
    if ! $all_done; then
        echo "$(date): $jobs_running jobs still running..."
        sleep 30
    fi
done

echo "All training jobs completed!"
echo "Results saved to ${LOGDIR} and logs saved to ${LOG_DIR}"
