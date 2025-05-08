#!/usr/bin/env python3

import os
import argparse
import subprocess
import time
import signal
import psutil
from typing import List, Dict


class GPUJobScheduler:
    def __init__(
        self,
        devices: List[int],
        num_experiments: int,
        max_jobs_per_gpu: int,
        dataset: str = "cifar10",
        epochs: int = 100,
        save_steps: int = 20,
        arch: str = "wrn28-2",
        logdir: str = "exp/cifar10",
        exclude_class: int = 0,
        log_dir: str = "logs"
    ):
        self.devices = devices
        self.num_experiments = num_experiments
        self.max_jobs_per_gpu = max_jobs_per_gpu
        self.dataset = dataset
        self.epochs = epochs
        self.save_steps = save_steps
        self.arch = arch
        self.logdir = logdir
        self.exclude_class = exclude_class
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Dictionary to track running jobs on each GPU
        self.gpu_jobs: Dict[int, Dict[int, subprocess.Popen]] = {gpu: {} for gpu in devices}
        
    def is_process_running(self, pid):
        """Check if a process is still running."""
        try:
            # Check if process is still running without sending any signal
            os.kill(pid, 0)
            return True
        except OSError:
            return False
        
    def get_available_gpu(self):
        """Find the GPU with the fewest running jobs that's below capacity."""
        gpu_job_counts = {}
        
        # Count active jobs on each GPU
        for gpu, jobs in self.gpu_jobs.items():
            # Filter out completed jobs
            active_jobs = {expid: proc for expid, proc in jobs.items() 
                          if proc.poll() is None}
            print(gpu, active_jobs)
            
            # Update jobs dictionary to only include active jobs
            self.gpu_jobs[gpu] = active_jobs
            gpu_job_counts[gpu] = len(active_jobs)

        print(gpu_job_counts)
        # Find least busy GPU that's below capacity
        available_gpus = [gpu for gpu, count in gpu_job_counts.items() 
                         if count < self.max_jobs_per_gpu]

        print(available_gpus)
        if not available_gpus:
            return None
            
        # Return GPU with fewest running jobs
        return min(available_gpus, key=lambda gpu: gpu_job_counts[gpu])
    
    def run_job(self, expid, gpu):
        """Start a training job on the specified GPU."""
        cmd = [
            "python3.10", "-u", "train.py",
            f"--dataset={self.dataset}",
            f"--epochs={self.epochs}",
            f"--save_steps={self.save_steps}",
            f"--arch", self.arch,
            f"--num_experiments", str(self.num_experiments),
            f"--expid", str(expid),
            f"--logdir", self.logdir,
        ]

        if self.exclude_class is not None:
            cmd.append(f"--exclude_class", str(self.exclude_class))
        
        # Set environment variables for GPU selection
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        
        # Open log file
        log_path = os.path.join(self.log_dir, f"log_{expid}")
        log_file = open(log_path, "w")
        
        print(f"Starting experiment {expid} on GPU {gpu} ({time.strftime('%H:%M:%S')})")
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True  # Ensures the process continues if script is interrupted
        )
        
        # Store the process in our tracking dictionary
        self.gpu_jobs[gpu][expid] = process
        
        return process
    
    def schedule_jobs(self):
        """Schedule all experiments across available GPUs."""
        print(f"Starting {self.num_experiments} training jobs across {len(self.devices)} GPUs {self.devices}")
        print(f"Maximum {self.max_jobs_per_gpu} concurrent jobs per GPU")
        
        expid = 0
        while expid < self.num_experiments:
            # Try to get an available GPU
            gpu = self.get_available_gpu()
            
            if gpu is not None:
                # Run job on this GPU
                self.run_job(expid, gpu)
                expid += 1
                
                # Small delay to stagger job starts
                time.sleep(2)
            else:
                print(f"All GPUs at capacity. Waiting for jobs to complete... ({time.strftime('%H:%M:%S')})")
                self.print_status()
                time.sleep(10)
        
        print("All jobs scheduled! Monitoring progress...")
    
    def print_status(self):
        """Print current status of jobs on each GPU."""
        for gpu in self.devices:
            active_jobs = sum(1 for proc in self.gpu_jobs[gpu].values() if proc.poll() is None)
            print(f"  GPU {gpu}: {active_jobs}/{self.max_jobs_per_gpu} jobs")
    
    def wait_for_completion(self):
        """Wait for all jobs to complete."""
        while True:
            all_done = True
            running_jobs = 0
            
            for gpu in self.devices:
                active_jobs = 0
                for expid, proc in list(self.gpu_jobs[gpu].items()):
                    if proc.poll() is None:  # Process still running
                        active_jobs += 1
                        running_jobs += 1
                        all_done = False
                print(f"GPU {gpu}: {active_jobs} jobs running")
            
            if all_done:
                print("All training jobs completed!")
                break
                
            print(f"{time.strftime('%H:%M:%S')}: {running_jobs} jobs still running...")
            time.sleep(30)
    
    def run(self):
        """Run the entire job scheduling process."""
        try:
            self.schedule_jobs()
            self.wait_for_completion()
            print(f"Results saved to {self.logdir} and logs saved to {self.log_dir}")
        except KeyboardInterrupt:
            print("Interrupted! Jobs will continue running in the background.")
            print("Use 'ps aux | grep train.py' to check running jobs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run distributed training jobs across multiple GPUs")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="GPU device IDs to use (default: 0 1 2 3)")
    parser.add_argument("--num_experiments", type=int, default=16,
                        help="Number of experiments to run (default: 16)")
    parser.add_argument("--max_jobs_per_gpu", type=int, default=2,
                        help="Maximum concurrent jobs per GPU (default: 2)")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Dataset name (default: cifar10)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--save_steps", type=int, default=20,
                        help="Save checkpoint frequency (default: 20)")
    parser.add_argument("--arch", type=str, default="wrn28-2",
                        help="Model architecture (default: wrn28-2)")
    parser.add_argument("--logdir", type=str, default="exp/cifar10",
                        help="Directory for experiment output (default: exp/cifar10)")
    parser.add_argument("--exclude_class", type=int, default=None,
                        help="Class to exclude (default: None)")
    parser.add_argument("--log_dir", type=str, default="all_logs/logs",
                        help="Directory for log files (default: logs)")
    
    args = parser.parse_args()

    #args.logdir += "_{:d}".format(args.exclude_class)
    #args.log_dir += "_{:d}".format(args.exclude_class)
    
    scheduler = GPUJobScheduler(
        devices=args.devices,
        num_experiments=args.num_experiments,
        max_jobs_per_gpu=args.max_jobs_per_gpu,
        dataset=args.dataset,
        epochs=args.epochs,
        save_steps=args.save_steps,
        arch=args.arch,
        logdir=args.logdir,
        exclude_class=args.exclude_class,
        log_dir=args.log_dir
    )
    
    scheduler.run()
