from __future__ import print_function, division
import h5py
import os
import sys
import numpy

gflags.DEFINE_integer("seq_len", 1500, "Sequence length.")
gflags.DEFINE_string("loss", "mse", "Loss to use for training.")

gflags.ADOPT_module_key_flags(helpers.videosampler)
gflags.ADOPT_module_key_flags(hantman_hungarian)
gflags.ADOPT_module_key_flags(flags.lstm_flags)
gflags.ADOPT_module_key_flags(arg_parsing)
gflags.ADOPT_module_key_flags(flags.cuda_flags)
gflags.ADOPT_module_key_flags(train)


def run_logits(opts, test_data):
    full_tic = time.time()
    fid.write("phase,timing\n")
    label_names = train_data["label_names"].value
    if valid_data is not None:
        data_files = [train_data, test_data, valid_data]
    else:
        data_files = [train_data, test_data, None]

    # setup output space.
    tic = time.time()
    _setup_templates(opts, data_files, label_names)
    toc = time.time()
    fid.write("output space,%f\n" % (toc - tic))
    print("Setup output space: %f" % (toc - tic))

    # create samplers for training/testing/validation.
    tic = time.time()
    samplers = _setup_samplers(opts, data_files)
    toc = time.time()
    fid.write("samplers,%f\n" % (toc - tic))
    print("Setup samplers: %f" % (toc - tic))

def main(argv):
    print(argv)
    opts = _setup_opts(argv)
    paths.setup_output_space(opts)
    if opts["flags"].cuda_device != -1:
        torch.cuda.set_device(opts["flags"].cuda_device)

    full_tic = time.time()
    with h5py.File(opts["flags"].test_file, "r") as test_data:
        run_training(opts, test_data)
    print("Training took: %d\n" % (time.time() - full_tic))


if __name__ == "__main__":
    main(sys.argv)

