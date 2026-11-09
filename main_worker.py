"""
main_worker.py
Manage multiple Python processes and their log files.
"""
import subprocess
import threading
import time
import os
import sys
from datetime import datetime
from pathlib import Path

class ProcessManager:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.processes = []  # Store process metadata.
        self.running = False
        self.log_dir.mkdir(exist_ok=True)
    
    def run_script(self, script_path, log_prefix="", env_vars=None, timeout=None):
        """
        Args:
            script_path: Path to the Python script.
            log_prefix: Prefix used for the log filename.
            env_vars: Dictionary of environment variables.
            timeout: Timeout in seconds; None disables the timeout.
        """
        if not Path(script_path).exists():
            raise FileNotFoundError(f"Script file does not exist: {script_path}")
        # Generate a unique log filename.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{log_prefix}_{timestamp}.log" if log_prefix else f"process_{timestamp}.log"
        log_filepath = self.log_dir / log_filename
        print(f"Starting script: {script_path}")
        print(f"Log file: {log_filepath}")
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        try:
            log_file = open(log_filepath, 'w', encoding='utf-8')
            log_file.write(f"=== Starting: {script_path} ===\n")
            log_file.write(f"Start time: {datetime.now()}\n\n")
            log_file.flush()
            process = subprocess.Popen([sys.executable, script_path],stdout=log_file,stderr=log_file,text=True,encoding='utf-8',env=env)
            process_info = {'process': process,'script': script_path,'log_file': log_file,'log_path': log_filepath,'start_time': datetime.now(),'timeout': timeout}
            self.processes.append(process_info)
            if timeout:
                monitor_thread = threading.Thread(target=self._monitor_process_timeout,args=(process_info,),daemon=True)
                monitor_thread.start()
            return process
        except Exception as e:
            print(f"Failed to start script: {e}")
            if 'log_file' in locals():
                log_file.close()
            raise
    
    def _monitor_process_timeout(self, process_info):
        """
        Monitor a process and terminate it when its timeout is reached.
        """
        time.sleep(process_info['timeout'])
        process = process_info['process']
        if process.poll() is None:  # The process is still running.
            print(f"Process timed out; terminating: {process_info['script']}")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    
    def run_multiple_scripts(self, script_path, count, log_prefix="process", env_vars=None):
        """
        Run multiple instances of the same Python script concurrently.
        
        Args:
            script_path: Path to the Python script.
            count: Number of instances to launch.
            log_prefix: Prefix used for log filenames.
            env_vars: Dictionary of environment variables.
        """
        print(f"Starting {count} instance(s) of {script_path}...")
        
        for i in range(count):
            # Add a unique identifier to each instance.
            instance_env = env_vars.copy() if env_vars else {}
            instance_env['PROCESS_INSTANCE'] = str(i + 1)
            
            try:
                self.run_script(
                    script_path=script_path,
                    log_prefix=f"{log_prefix}_{i+1}",
                    env_vars=instance_env
                )
                # Stagger launches slightly to avoid starting every process at once.
                time.sleep(0.1)
            except Exception as e:
                print(f"Failed to start instance {i+1}: {e}")
        
        print(f"Started {count} instance(s)")
    
    def run_specific_script_with_timeout(self, script_path, timeout_seconds, log_prefix="specific", env_vars=None):
        """
        Run a specific script for a fixed period.
        
        Args:
            script_path: Path to the Python script.
            timeout_seconds: Runtime in seconds.
            log_prefix: Prefix used for the log filename.
            env_vars: Dictionary of environment variables.
        """
        print(f"Running {script_path} for {timeout_seconds} seconds...")
        
        process = self.run_script(
            script_path=script_path,
            log_prefix=log_prefix,
            env_vars=env_vars,
            timeout=timeout_seconds
        )
        
        return process
    
    def wait_all(self):
        """
        Wait for all managed processes to finish.
        """
        print("Waiting for all processes to finish...")
        
        for process_info in self.processes:
            try:
                process_info['process'].wait()
                print(f"Process finished: {process_info['script']}")
            except Exception as e:
                print(f"Error while waiting for process: {e}")
        
        # Close all log files.
        for process_info in self.processes:
            if process_info['log_file']:
                process_info['log_file'].close()
    
    def get_status(self):
        """
        Return the status of every managed process.
        """
        status = []
        for i, process_info in enumerate(self.processes):
            process = process_info['process']
            returncode = process.poll()
            
            status.append({
                'index': i + 1,
                'script': process_info['script'],
                'status': 'running' if returncode is None else f'finished (exit code: {returncode})',
                'log_file': str(process_info['log_path']),
                'start_time': process_info['start_time']
            })
        
        return status
    def stop_all(self):
        """
        Stop all managed processes.
        """
        print("Stopping all processes...")
        for process_info in self.processes:
            process = process_info['process']
            if process.poll() is None:  # The process is still running.
                process.terminate()
                print(f"Terminated process: {process_info['script']}")
        # Close all log files.
        for process_info in self.processes:
            if process_info['log_file']:
                process_info['log_file'].close()

def main():
    """Parse command-line arguments and run the requested processes."""
    import argparse
    parser = argparse.ArgumentParser(description='Python process manager')
    parser.add_argument('--multiple-script', type=str, help='Path to a Python script to run multiple times')
    parser.add_argument('--count', type=int, default=3, help='Number of instances to run')
    parser.add_argument('--specific-script', type=str, help='Path to a Python script to run for a fixed period')
    parser.add_argument('--timeout', type=int, default=10, help='Runtime of the specific script in seconds')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    args = parser.parse_args()
    manager = ProcessManager(log_dir=args.log_dir)
    try:
        if args.multiple_script:
            manager.run_multiple_scripts(
                script_path=args.multiple_script,
                count=args.count,
                log_prefix="worker"
            )
        if args.specific_script:
            manager.run_specific_script_with_timeout(
                script_path=args.specific_script,
                timeout_seconds=args.timeout,
                log_prefix="specific"
            )
        # Show usage examples when no script is specified.
        if not args.multiple_script and not args.specific_script:
            print("Specify at least one script to run:")
            print("Example 1: run three worker.py instances")
            print("  python main_worker.py --multiple-script worker.py --count 3")
            print("\nExample 2: run specific.py for 10 seconds")
            print("  python main_worker.py --specific-script specific.py --timeout 10")
            print("\nExample 3: run multiple collectors and one timed trainer")
            print("  python main_worker.py --multiple-script collect.py --count 4 --specific-script train.py --timeout 7200")
            return
        # Display process status.
        print("\nCurrent process status:")
        status = manager.get_status()
        for s in status:
            print(f"  {s['index']}. {s['script']} - {s['status']}")
            print(f"     Log: {s['log_file']}")
        print("\nWaiting for all processes to finish...")
        manager.wait_all()
        print("\nAll processes have finished")

    except KeyboardInterrupt:
        print("\nInterrupt received; stopping all processes...")
        manager.stop_all()
    except Exception as e:
        print(f"Runtime error: {e}")
        manager.stop_all()

if __name__ == "__main__":
    main()
