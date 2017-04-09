import subprocess

start = 0
end = 1000
step = 10

for lower in range(start, end, step):
    command = ['sbatch', '-c', '2', '--time=1-12:0', '-J', str(lower), 'generate_worlds.py', '--lower', str(lower), '--num_worlds', str(step)]
    subprocess.Popen( command )