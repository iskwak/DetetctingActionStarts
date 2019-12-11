# pdb is stupid
import sys
sys.path.append('./')

# import threaded_hungarian_mouse
import cached_thumos_matching

DEBUG = True
if DEBUG is True:
    train_file  = "/groups/branson/bransonlab/kwaki/data/thumos14/h5data/debug.hdf5"
    test_file = "/groups/branson/bransonlab/kwaki/data/thumos14/h5data/debug.hdf5"
    valid_file = ''
else:
    train_file  = "/groups/branson/bransonlab/kwaki/data/thumos14/h5data/train.hdf5"
    test_file = "/groups/branson/bransonlab/kwaki/data/thumos14/h5data/debug.hdf5"
    valid_file = ''

arg_string = [
    # 'threaded_hungarian_mouse_test.py',
    'cached_thumos_matching.py',
    '--train_file', train_file,
    '--test_file', test_file,
    # '--valid_file', valid_file,
    '--learning_rate', '0.00001',
    '--mini_batch', '5',
    '--total_epochs', '6',
    '--display_dir', '/groups/branson/bransonlab/kwaki/data/thumos14/videos',
    '--video_dir', '/groups/branson/bransonlab/kwaki/data/thumos14/videos',
    '--cuda_device', '0',
    '--reweight',
    '--normalize',
    '--save_iterations', '2',
    '--update_iterations', '2',
    '--out_dir', '/nrs/branson/kwaki/outputs/tests/20190719_test',
    '--hantman_arch', 'bidirconcat',
    '--lstm_hidden_dim', '256',
    '--lstm_num_layers', '2',
    '--feat_keys', 'canned_i3d_rgb_64',
    '--label_key', 'end_labels',
    '--nouse_pool',
    # '--loss', 'weighted_mse',
    '--loss', 'wasserstein',
    '--hantman_perframe_weight', '0.9',
    '--anneal_type', 'line_step',
    '--perframe_stop', '0.5',
    '--seq_len', '-1',
    '--cached',
    '--label_smooth_win', '59',
    '--label_smooth_std', '8',
]

cached_thumos_matching.main(arg_string)
