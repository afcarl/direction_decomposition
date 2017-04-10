optspace=1000
iters=1000
rank=20
path=logs/

for worlds in 5 10 15 25 40 50 75 100;
do
    sbatch -c 2 --gres=gpu:1 -J ${worlds} --time=3-12:0 main.py --save_path ${path}${rank}_${worlds} --optspace_iters ${optspace} --iters ${iters} --rank ${rank} --num_worlds ${worlds}
done