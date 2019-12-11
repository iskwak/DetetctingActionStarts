# pdb is stupid
import sys
sys.path.append('./')

# import threaded_hungarian_mouse as app
import load_hungarian_mouse as app

DEBUG = True

if DEBUG is True:
    train_file = '/groups/branson/bransonlab/kwaki/data/thumos14/h5data/debug.hdf5'
    test_file = '/groups/branson/bransonlab/kwaki/data/thumos14/h5data/debug.hdf5'
    valid_file = ''
else:
    train_file = '/groups/branson/bransonlab/kwaki/data/thumos14/h5data/train.hdf5'
    test_file = '/groups/branson/bransonlab/kwaki/data/thumos14/h5data/test.hdf5'
    valid_file = ''

arg_string = [
    # 'threaded_hungarian_mouse_test.py',
    'load_hungarian_mouse.py',
    '--train_file', train_file,
    '--test_file', test_file,
    # '--valid_file', valid_file,
    '--learning_rate', '0.0001',
    '--mini_batch', '10',
    '--total_epochs', '500',
    '--display_dir', '/groups/branson/bransonlab/kwaki/data/thumos14/videos',
    '--video_dir', '/groups/branson/bransonlab/kwaki/data/thumos14/videos',
    '--cuda_device', '0',
    '--reweight',
    '--normalize',
    '--save_iterations', '10',
    '--update_iterations', '10',
    # '--out_dir', '/nrs/branson/kwaki/outputs/tests/20190514_test3',
    '--out_dir', '/nrs/branson/kwaki/outputs/thumos14/test/beep',
    '--hantman_arch', 'bidirconcat',
    '--lstm_hidden_dim', '256',
    '--lstm_num_layers', '2',
    # '--feat_keys', 'rgb_i3d_view1_fc,rgb_i3d_view2_fc',
    '--feat_keys', 'canned_i3d_rgb_64_past',
    # '--feat_keys', 'canned_i3d_rgb_64',
    '--nouse_pool',
    # '--loss', 'weighted_mse',
    '--loss', 'hungarian',
    # '--loss', 'wasserstein',
    '--anneal_type', 'exp_step',
    '--seq_len', '4000',
    '--label_key', 'labels',
    '--label_smooth_win', '59',
    '--label_smooth_std', '8',
    # '--hantman_perframe_weight', '0.5',
    # '--perframe_stop', '0.5',
    '--hantman_perframe_weight', '0.99',
    '--perframe_decay', '0.9',
    '--perframe_decay_step', '5',
    '--perframe_stop', '0.5',
    '--hantman_tp', '4.0',
    '--hantman_fp', '1.0',
    '--hantman_fn', '2.0'
]

app.main(arg_string)
