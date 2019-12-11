"""Helper for setting up parameters for paramssweep_busb"""


def hungarianmouse_setup():
    """Setup the hungarian mouse parameters."""
    base_command = (
        'python load_hungarian_mouse.py '
        # 'python threaded_hungarian_mouse.py '
        '--train_file %train_file% '
        '--test_file %test_file% '
        # '--valid_file %val_file% '
        # '--hantman_arch bidirconcat '
        '--hantman_arch %arch% '
        '--feat_keys %feat_keys% '
        '--label_key %label_key% '
        '--mini_batch 10 '
        '--total_epochs 400 '
        # '--display_dir /nrs/branson/kwaki/data/hantman_mp4 '
        # '--video_dir  /nrs/branson/kwaki/data/hantman_pruned '
        '--display_dir /groups/branson/bransonlab/kwaki/data/thumos14/videos '
        '--video_dir /groups/branson/bransonlab/kwaki/data/thumos14/videos '
        '--cuda_device 0 '
        '--reweight '
        '--save_iterations 50 '
        '--update_iterations 400 '
        '--out_dir %out_dir% '
        '--lstm_num_layers 1 '
        '--use_pool '
        '--learning_rate %learning_rate%  '
        '--lstm_hidden_dim 256 '
        '--normalize '
        '--anneal_type %anneal_type% '
        '--loss %loss% '
        '--hantman_perframe_weight %perframe% '
        '--perframe_decay %decay% '
        '--perframe_decay_step %decay_step% '
        '--perframe_stop %perframe_stop% '
        # '--seq_len=1500 '
        '--seq_len=2500 '
        '--hantman_tp 4.0 '
        '--hantman_fp 1.0 '
        '--hantman_fn 2.0 '
        '--label_smooth_win 59 '
        '--label_smooth_std 8 '
    )

    main_params = {
        'train_file': '/groups/branson/bransonlab/kwaki/data/thumos14/h5data/train.hdf5',
        'test_file': '/groups/branson/bransonlab/kwaki/data/thumos14/h5data/test.hdf5',
        'feat_keys': 'canned_i3d_rgb_64',
        # 'feat_keys': 'canned_i3d_rgb_64_past',
    }
    # mse wasser
    # tunable_params = {
    #     'anneal_type': ['none'],
    #     'decay': [0.9],
    #     'perframe': [0.5],
    #     'decay_step': [1],
    #     'perframe_stop': [0.5],
    #     'learning_rate': [0.00001],
    #     # 'loss': ['wasserstein'],
    #     # 'perframe_stop': [0.5],
    #     # 'loss': ['wasserstein'],
    #     # 'loss': ['weighted_mse'],
    #     'loss': ['weighted_mse', 'wasserstein'],
    #     # 'loss': ['weighted_mse', 'wasserstein', 'hungarian']
    #     'arch': ['concat', 'bidirconcat'],
    #     'label_key': ['labels', 'end_labels']
    # }
    # tunable_params = {
    #     'anneal_type': ['exp_step'],
    #     'decay': [0.9],
    #     'perframe': [0.99],
    #     'decay_step': [5],
    #     'perframe_stop': [0.5],
    #     'learning_rate': [0.00001],
    #     # 'loss': ['wasserstein'],
    #     # 'perframe_stop': [0.5],
    #     # 'loss': ['wasserstein'],
    #     # 'loss': ['weighted_mse'],
    #     'loss': ['wasserstein'],
    #     # 'loss': ['weighted_mse', 'wasserstein', 'hungarian']
    #     'arch': ['concat', 'bidirconcat'],
    #     'label_key': ['labels', 'end_labels']
    # }
    # tunable_params = {
    #     'anneal_type': ['exp_step'],
    #     'decay': [0.9],
    #     'perframe': [0.99],
    #     'decay_step': [5],
    #     'perframe_stop': [0.5],
    #     'loss': ['hungarian'],
    #     'hantman_tp': [4.0],
    #     'hantman_fp': [1.0],
    #     'hantman_fn': [2.0],
    #     'learning_rate': [0.00001],
    #     'arch': ['concat', 'bidirconcat'],
    #     'label_key': ['labels', 'end_labels']
    # }
    tunable_params = {
        'anneal_type': ['none'],
        'decay': [0.9],
        'perframe': [0.75],
        'decay_step': [5],
        'perframe_stop': [0.75],
        'loss': ['wasserstein'],
        'learning_rate': [0.00001],
        'arch': ['concat', 'bidirconcat'],
        'label_key': ['labels']
    }

    # output_dir = '/nrs/branson/kwaki/outputs/thumos14/canned_64_past_bigger_win'
    output_dir = '/nrs/branson/kwaki/outputs/thumos14/canned_64_bigger_win'

    return base_command, main_params, tunable_params, output_dir


    # tunable_params = {
    #     'learning_rate': [0.0001],
    #     'hidden_dim': [256],
    #     'anneal_type': ['exp_step'],
    #     'decay': [0.9],
    #     'perframe': [0.99],
    #     'decay_step': [1],
    #     # 'loss': ['weighted_mse']
    #     # 'loss': ['hungarian']
    #     # 'loss': ['wasserstein']
    #     'loss': ['hungarian', 'weighted_mse']
    #     # 'arch': ['concat', 'bidirconcat']
    # }
    # weighted mse params
    # tunable_params = {
    #     # 'learning_rate': [0.001, 0.0001],
    #     # 'hidden_dim': [128, 256],
    #     'learning_rate': [0.001, 0.0001],
    #     'hidden_dim': [128, 256],
    #     # 'anneal': [0.5, 2],
    #     # 'decay': [0.5]
    #     # 'arch': ['concat', 'bidirconcat']
    # }


def mpii_setup():
    """Setup the mpii parameters."""
    base_command = (
        'python mpii.py '
        '--train_file %train_file% '
        '--test_file %test_file% '
        '--arch %arch% '
        '--feat_keys %feat_keys% '
        '--out_dir %out_dir% '
        '--learning_rate %learning_rate% '
        '--lstm_hidden_dim %lstm_hidden_dim% '
        '--image_dir /localhome/kwaki/frames '
        '--cuda_device 0 '
        '--hantman_mini_batch=10 '
        '--hantman_perframeloss=WEIGHTED_MSE '
        '--seq_len=5000 '
        '--total_epochs=100 '
        '--hantman_perframe_weight=100.0 '
        '--hantman_struct_weight=1.0 '
        '--hantman_tp=10.0 '
        '--hantman_fp=0.25 '
        '--hantman_fn=20.0 '
        '--reweight --normalize'
    )

    # main parameters
    # 'val_file': '/nrs/branson/kwaki/data/20180328_mpiicooking2',
    main_params = {
        'train_file': '/nrs/branson/kwaki/data/20180328_mpiicooking2/temp_data/hdf5/train.hdf5',
        'test_file': '/nrs/branson/kwaki/data/20180328_mpiicooking2/temp_data/hdf5/test.hdf5',
        'arch': 'bidirconcat',
        'feat_keys': 'vgg',
        'out_dir': '',
        'learning_rate': '',
        'lstm_hidden_dim': ''
    }

    # learning_rates = [0.01, 0.001, 0.0001]
    # hidden_dims = [64, 128, 256, 512]
    learning_rates = [0.0001]
    hidden_dims = [256]

    output_dir = '/nrs/branson/kwaki/outputs/20180403_mpii_sweep_test'
    # output_dir = '/nrs/branson/kwaki/outputs/20180411_mpii_tests'

    return base_command, main_params, learning_rates, hidden_dims, output_dir


def hantman_3dconv():
    base_command = (
        'python hantman_3dconv.py '
        '--train_file %train_file% '
        '--test_file %test_file% '
        '--valid_file %valid_file% '
        '--out_dir %out_dir% '
        '--learning_rate %learning_rate% '
        '--hantman_mini_batch=%hantman_mini_batch% '
        '--total_epochs=%total_epochs% '
        '--display_dir /nrs/branson/kwaki/data/hantman_mp4 '
        '--video_dir /nrs/branson/kwaki/data/hantman_pruned '
        '--cuda_device 0 '
        '--reweight '
        '--save_iterations 30 '
        '--update_iterations 30 '
        '--frames %frames% '
    )

    # main parameters
    # 'val_file': '/nrs/branson/kwaki/data/20180328_mpiicooking2',
    main_params = {
        'train_file': '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_REDO_train.hdf5',
        'valid_file': '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_REDO_valid.hdf5',
        'test_file': '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_REDO_test.hdf5',
        'hantman_mini_batch': '2',
        'total_epochs': 60,
        'frames': '"-5 -4 -3 -2 -1 0 1 2 3 4 5"'
    }

    # learning_rates = [0.01, 0.001, 0.0001]
    # hidden_dims = [64, 128, 256, 512]
    # learning_rates = [0.001, 0.0001]
    # hidden_dims = [0]
    tunable_params = {
        'learning_rate': [0.001, 0.0001],
    }

    output_dir = '/nrs/branson/kwaki/outputs/20190110_REDO_3dconvforward_conv4'
    # output_dir = '/nrs/branson/kwaki/outputs/20180411_mpii_tests'

    return base_command, main_params, tunable_params, output_dir


def hantman_2dconv():
    base_command = (
        'python feedforward.py '
        '--train_file %train_file% '
        '--test_file %test_file% '
        '--valid_file %valid_file% '
        '--out_dir %out_dir% '
        '--learning_rate %learning_rate% '
        '--hantman_mini_batch=%hantman_mini_batch% '
        '--total_epochs=%total_epochs% '
        '--display_dir /nrs/branson/kwaki/data/hantman_mp4 '
        '--video_dir /nrs/branson/kwaki/data/hantman_pruned '
        '--cuda_device 0 '
        '--reweight '
        '--use_track %use_track% '
    )

    # main parameters
    # 'val_file': '/nrs/branson/kwaki/data/20180328_mpiicooking2',
    main_params = {
        'hantman_mini_batch': '10',
        'total_epochs': 1000,
        'learning_rate': 0.0001,
        'train_file': '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_REDO_train.hdf5',
        'test_file': '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_REDO_test.hdf5',
        'valid_file': '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_REDO_valid.hdf5',
    }

    tunable_params = {
        "use_track": [0, 1]
    }

    # learning_rates = [0.01, 0.001, 0.0001]
    # hidden_dims = [64, 128, 256, 512]

    output_dir = '/nrs/branson/kwaki/outputs/20181105_2dconvforward_redo_REDO'

    return base_command, main_params, tunable_params, output_dir
    # return base_command, main_params, learning_rates, hidden_dims, output_dir


def feature_creation():
    base_command = (
        'python feedforward_eval.py '
        '--train_file %train_file% '
        '--test_file %test_file% '
        '--valid_file %valid_file% '
        '--load_network %network% '
        '--out_dir %out_dir% '
    )

    main_params = {
        'train_file': '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_REDO_train.hdf5',
        'test_file': '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_REDO_test.hdf5',
        'valid_file': '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_REDO_valid.hdf5',
    }

    # use the output dir to help name the parameters
    tunable_params = {
        'split': 'REDO'
    }

    output_dir = '/nrs/branson/kwaki/data/features/hantman_2dconv'

    return base_command, main_params, tunable_params, output_dir
