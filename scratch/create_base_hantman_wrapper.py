""""Wrapper for helpers.create_base_hantman_hdf"""
import helpers.create_base_hantman_hdf as create_base_hantman_hdf

arg_string = [
    "helpers/create_base_hantman_hdf.py",
    "--input_dir", "/media/drive1/data/hantman_processed/hdf5_data",
    "--out_dir", "/media/drive1/data/hantman_processed/20180708_base_hantman"
    # "--out_dir", "/nrs/branson/kwaki/data/20180708_base_hantman"
]

opts = create_base_hantman_hdf._setup_opts(arg_string)
create_base_hantman_hdf.main(opts)
