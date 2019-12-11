# pdb is stupid
import sys
sys.path.append('./')

# import threaded_hungarian_mouse
import eval_hungarian_mouse

train_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M174_train.hdf5'
test_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M174_test.hdf5'
valid_file = ''

arg_string = [
    # 'threaded_hungarian_mouse_test.py',
    'eval_hungarian_mouse.py',
    '--train_file', train_file,
    '--test_file', test_file,
    # '--valid_file', valid_file,
    '--learning_rate', '0.00001',
    '--mini_batch', '10',
    '--total_epochs', '400',
    '--display_dir', '/nrs/branson/kwaki/data/hantman_mp4',
    '--video_dir', '/nrs/branson/kwaki/data/hantman_pruned',
    '--cuda_device', '0',
    '--reweight',
    '--normalize',
    '--save_iterations', '10',
    '--update_iterations', '10',
    '--out_dir', '/nrs/branson/kwaki/outputs/M134/canned/weighted',
    '--hantman_arch', 'bidirconcat',
    '--lstm_hidden_dim', '256',
    '--lstm_num_layers', '2',
    # '--feat_keys', 'rgb_i3d_view1_fc,rgb_i3d_view2_fc',
    # '--feat_keys', 'hoghof',
    '--feat_keys', 'canned_i3d_rgb_front,canned_i3d_rgb_side,canned_i3d_flow_front,canned_i3d_flow_side',
    '--nouse_pool',
    '--loss', 'weighted_mse',
    # '--loss', 'hungarian',
    # '--loss', 'wasserstein',
    '--hantman_perframe_weight', '0.9',
    '--anneal_type', 'line_step',
    '--perframe_stop', '0.5',
    '--model', '/nrs/branson/kwaki/outputs/M134/canned_i3d/20190401-perframe_stop_0.5-perframe_0.99-loss_wasserstein-decay_step_1-decay_0.9-anneal_type_exp_step/networks/network.pt'
    # '--model', '/nrs/branson/kwaki/outputs/M134/hoghof/20190404-perframe_stop_0.5-perframe_0.99-loss_hungarian-hantman_tp_4.0-hantman_fp_1.0-hantman_fn_20.0-decay_step_5-decay_0.9-anneal_type_exp_step/networks/41600/network.pt'
    # '--model', '/nrs/branson/kwaki/outputs/M147/hoghof/20190405-perframe_stop_0.5-perframe_0.99-loss_hungarian-learning_rate_1e-05-hantman_tp_4.0-hantman_fp_1.0-hantman_fn_2.0-decay_step_5-decay_0.9-anneal_type_exp_step/networks/88800/network.pt'
    # '--model', '/nrs/branson/kwaki/outputs/M173/hoghof/20190404-perframe_stop_0.5-perframe_0.99-loss_hungarian-hantman_tp_4.0-hantman_fp_1.0-hantman_fn_20.0-decay_step_5-decay_0.9-anneal_type_exp_step/networks/35600/network.pt'
    # '--model', '/nrs/branson/kwaki/outputs/M174/hoghof/20190405-perframe_stop_0.5-perframe_0.99-loss_hungarian-learning_rate_1e-05-hantman_tp_4.0-hantman_fp_1.0-hantman_fn_2.0-decay_step_5-decay_0.9-anneal_type_exp_step/networks/78000/network.pt'
]

eval_hungarian_mouse.main(arg_string)
