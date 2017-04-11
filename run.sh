train_path=data/train_10/
test_path=data/test_10/
optspace_iters=2000
phi_iters=2000
psi_iters=5000
rank=20
path=logs/test_relu_mapdim10_
sanity_check=2


for worlds in 20 40 60 80 100;
do
    fullpath=${path}rank${rank}_worlds${worlds}_opt${optspace_iters}_phi${phi_iters}_psi${psi_iters}
    echo ${worlds}: ${fullpath}
    sbatch -c 2 --gres=gpu:1 -J ${worlds} --time=3-12:0 main.py \
        --data_path ${train_path} --test_path ${test_path} --save_path ${fullpath} \
        --optspace_iters ${optspace_iters} --phi_iters ${phi_iters} --psi_iters ${psi_iters} \
        --rank ${rank} --num_worlds ${worlds} --sanity_check ${sanity_check}
done