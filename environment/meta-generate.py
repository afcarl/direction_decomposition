import subprocess

save_path = '../data/test_10/'
vis_path = '../data/vis_10/'
start = 0
end = 1000
step = 10
dim = 10

for lower in range(start, end, step):
    command = [ 'sbatch', '-c', '2', '--time=1-12:0', '-J', str(lower), 'generate_worlds.py', \
                '--lower', str(lower), '--num_worlds', str(step), '--dim', str(dim), \
                '--save_path', save_path, '--vis_path', vis_path]
    # print command
    subprocess.Popen( command )