"""Test the bidirectional lstm model."""
import numpy
import models.hantman_bilstm as hantman_bilstm
# import theano
# import theano.tensor as T

rng = numpy.random.RandomState(456)
eps = numpy.finfo(numpy.float32).eps

opts = dict()
opts["rng"] = rng
opts["filename"] = ""
opts["total_iterations"] = 1000
opts["num_train"] = 100
opts["num_valid"] = 100
opts["num_hidden"] = 50
opts["mini_batch_size"] = 100
# opts["seq_length"] = g_seq_length
opts["learning_rate"] = 0.001
opts["decay_step"] = 0
opts["decay_rate"] = 0.5
opts["weightdecay"] = 0.0001
opts["num_classes"] = 6
opts["out_dir"] = ""
opts["update_iter"] = 1000
opts["image_dir"] = None
opts["load_network"] = None

opts["num_input"] = 10

params_dict, fwd_names, bwd_names = hantman_bilstm.create_network_weights(opts)
hantman_bilstm.create_network(opts, params_dict, fwd_names, bwd_names)


# def accum(xf, xr, hf, hr):
#     hf_t = hf + xf
#     hr_t = hr + xr
#
#     return hf_t, hr_t
#
# xf = T.matrix("xf")
# xr = T.matrix("xr")
#
# init_hf = T.alloc(numpy.cast[theano.config.floatX](0), 4)
# init_hr = T.alloc(numpy.cast[theano.config.floatX](0), 4)
#
# [hf, hr], _ = theano.scan(
#     accum,
#     sequences=[xf, xr],
#     outputs_info=[init_hf, init_hr])
#
# func = theano.function([xf, xr], [hf, hr[::-1]])
#
# xf_data = numpy.asarray(
#     [[1, 1, 1, 1],
#      [2, 2, 2, 2],
#      [3, 3, 3, 3],
#      [4, 4, 4, 4],
#      [5, 5, 5, 5]], dtype="float32")
# xr_data = numpy.asarray(
#     [[1, 1, 1, 1],
#      [2, 2, 2, 2],
#      [3, 3, 3, 3],
#      [4, 4, 4, 4],
#      [5, 5, 5, 5]], dtype="float32")
#
# print func(xf_data, xr_data)
