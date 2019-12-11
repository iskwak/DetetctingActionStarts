"""Helper script to run hungarian_lstm.py"""
import hungarian_lstm

arg_string = [
    'hungarian_lstm.py',
    '--train_file', '/nrs/branson/kwaki/data/20180619_jigsaw_base/Suturing_1_Out_train.hdf5',
    '--test_file', '/nrs/branson/kwaki/data/20180619_jigsaw_base/Suturing_1_Out_test.hdf5',
    '--out_dir', '/nrs/branson/kwaki/outputs/20180814_jigsaw_test',
    '--mini_batch', '5',
    '--display_dir', '/nrs/branson/kwaki/data/jigsaw_mp4',
    '--cuda_device', '2',
    '--save_iterations', '10',
    '--update_iterations', '10',
    '--total_epochs', '200',
    '--seq_len', '4000',
    '--split', 'split_1',
    '--feat_keys', 'hog_eq,hof_eq',
    '--learning_rate', '0.001',
    '--lstm_hidden_dim', '256',
    '--arch', 'bidir',
    '--normalize',
    '--reweight',
]

hungarian_lstm.main(arg_string)
