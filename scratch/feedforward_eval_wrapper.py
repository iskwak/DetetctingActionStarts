"""Wrapper for feedforward testing."""
import feedforward_eval


arg_string = [
    'feedforward_eval.py',
    '--train_file', '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M134_train.hdf5',
    '--test_file', '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M134_test.hdf5',
    '--valid_file', '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M134_valid.hdf5',
    '--feature_dir', '/nrs/branson/kwaki/data/features/hantman_2dconv/20181103_splitM134',
    '--learning_rate', '0.0001',
    '--hantman_mini_batch', '10',
    '--total_epochs', '100',
    '--display_dir', '/nrs/branson/kwaki/data/hantman_mp4',
    '--video_dir', '/nrs/branson/kwaki/data/hantman_pruned',
    '--cuda_device', '2',
    '--reweight',
    '--save_iterations', '10',
    '--update_iterations', '10',
    '--out_dir', '/nrs/branson/kwaki/outputs/20181104_m134_features',
    '--load_network', '/nrs/branson/kwaki/outputs/20181105_2dconvforward_redo_M134/20181107-use_track_1/networks/1036494/network.pt'
]


feedforward_eval.main(arg_string)



arg_string = [
    'feedforward_eval.py',
    '--train_file', '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M173_train.hdf5',
    '--test_file', '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M173_test.hdf5',
    '--valid_file', '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M173_valid.hdf5',
    '--feature_dir', '/nrs/branson/kwaki/data/features/hantman_2dconv/20181103_splitM173',
    '--learning_rate', '0.0001',
    '--hantman_mini_batch', '10',
    '--total_epochs', '100',
    '--display_dir', '/nrs/branson/kwaki/data/hantman_mp4',
    '--video_dir', '/nrs/branson/kwaki/data/hantman_pruned',
    '--cuda_device', '0',
    '--reweight',
    '--save_iterations', '10',
    '--update_iterations', '10',
    '--out_dir', '/nrs/branson/kwaki/outputs/20181112_m173_features',
    '--load_network', '/nrs/branson/kwaki/outputs/20181105_2dconvforward_redo_M173/20181105-use_track_1/networks/87365/network.pt'
]




arg_string = [
    'feedforward_eval.py',
    '--train_file', '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M147_train.hdf5',
    '--test_file', '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M147_test.hdf5',
    '--valid_file', '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M147_valid.hdf5',
    '--feature_dir', '/nrs/branson/kwaki/data/features/hantman_2dconv/20181103_splitM147',
    '--learning_rate', '0.0001',
    '--hantman_mini_batch', '10',
    '--total_epochs', '100',
    '--display_dir', '/nrs/branson/kwaki/data/hantman_mp4',
    '--video_dir', '/nrs/branson/kwaki/data/hantman_pruned',
    '--cuda_device', '0',
    '--reweight',
    '--save_iterations', '10',
    '--update_iterations', '10',
    '--out_dir', '/nrs/branson/kwaki/outputs/20181104_m147_features',
    '--load_network', '/nrs/branson/kwaki/outputs/20181105_2dconvforward_redo_M147/20181107-use_track_1/networks/1365636/network.pt'
]


arg_string = [
    'feedforward_eval.py',
    '--train_file', '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M174_train.hdf5',
    '--test_file', '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M174_test.hdf5',
    '--valid_file', '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M174_valid.hdf5',
    '--feature_dir', '/nrs/branson/kwaki/data/features/hantman_2dconv/20181103_splitM174',
    '--learning_rate', '0.0001',
    '--hantman_mini_batch', '10',
    '--total_epochs', '100',
    '--display_dir', '/nrs/branson/kwaki/data/hantman_mp4',
    '--video_dir', '/nrs/branson/kwaki/data/hantman_pruned',
    '--cuda_device', '0',
    '--reweight',
    '--save_iterations', '10',
    '--update_iterations', '10',
    '--out_dir', '/nrs/branson/kwaki/outputs/20181104_m174_features',
    '--load_network', '/nrs/branson/kwaki/outputs/20181105_2dconvforward_redo_M174/20181107-use_track_1/networks/1147000/network.pt'
]
