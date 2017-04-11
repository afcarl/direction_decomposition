import os, pipeline

def check_factorization(recon, true, save_path, num_vis=10):
    num_states = true.shape[0]
    state_size = true.shape[1]

    checked_states = min(num_vis * state_size, num_states)

    for i in range(0, checked_states, state_size):
        for j in range(state_size):
            ## check if observed
            if true[i][j] != 0:
                for col_offset in range(10):
                    j_new = (j + col_offset) % state_size
                    fullpath = os.path.join(save_path, str(i) + '_' + str(j_new) + '.png')
                    true_values = true[i:i+state_size, j_new]
                    pred_values = recon[i:i+state_size, j_new]

                    if col_offset == 0:
                        title = 'observed'
                    else:
                        title = 'extrapolate: ' + str(col_offset) + ' right'

                    pipeline.visualize_value_map(pred_values, true_values, fullpath, title=title)

