"""Helper with hungarain matching tools."""
import numpy
import scipy.optimize
# import helpers.post_processing as post_processing


def apply_hungarian(ground_idx, predict_idx, val_threshold=0.7, dist_threshold=10):
    """Applies hungarian matching on two sequences."""
    # First apply non max suppression on the two sequences.
    # ground_suppress, ground_idx = post_processing.nonmax_suppress(
    #     ground_truth, val_threshold=val_threshold)
    # predict_suppress, predict_idx = post_processing.nonmax_suppress(
    #     predictions, val_threshold=val_threshold)

    # create a distance matrix for the matching to use
    # dist_mat = create_frame_dists(predict_vals, target_vals)
    dist_mat = create_frame_dists(ground_idx, predict_idx)

    num_target = len(ground_idx)
    num_predict = len(predict_idx)
    # import pdb; pdb.set_trace()
    # augment the distance matrix for hungarian matching.
    # This gives the matching algorithm dummy variables to match to if the
    # closest point is greater than the distance threshold distance.
    # Put the dummy nodes first... this will allow elements that tie with the
    # distance threshold to go first...
    dist_mat = numpy.concatenate(
        [numpy.zeros((num_target, num_target)) + dist_threshold, dist_mat],
        axis=1)
    match_rows, match_cols = scipy.optimize.linear_sum_assignment(dist_mat)

    # figure out the matched rows/cols and the TP's, FP's, and FN's.
    fps = list(range(num_predict))
    tps = []
    # matched_predicts = []
    for i in range(len(match_rows)):
        # need to adjust the indexing into the match array. For dealing with
        # ties, added a buffer of dummy nodes at the start of the dist_mat.
        # Remove buffer for saving purposes.
        match_col_adj = match_cols[i] - num_target
        # figure out if the match is a TP or FP.
        if match_cols[i] >= num_target:
            # this must be a true positive. It was matched by the hungarin
            # algorithm, and isn't a dummy node.
            # tps.append(
            #     (match_rows[i], match_col_adj, ground_idx[i], predict_idx[match_col_adj])
            # )
            tps.append(
                (ground_idx[i], predict_idx[match_col_adj])
            )

            # as columns are used (ie predicts are matched), remove them from
            # the false positive list.
            fps.remove(match_col_adj)
    # after figuring out the true pos and false pos. Compute the number of
    # false neg. This is the number of targets minus the true positives.
    num_fn = num_target - len(tps)

    # convert the fps from indexing into predict_idx, got just being the values
    # stored in predict_idx.
    fps = [
        predict_idx[fps_idx] for fps_idx in fps
    ]

    match_dict = {
        "tps": tps,
        "fps": fps,
        "num_fn": num_fn
    }

    return match_dict, dist_mat


def create_frame_dists(frames1, frames2):
    """Helper function to create frame distances."""
    # total = labelled.shape[0]
    dist_mat = numpy.zeros((len(frames1), len(frames2)))
    for i in range(len(frames1)):
        for j in range(len(frames2)):
            dist_mat[i, j] = numpy.abs(frames1[i] - frames2[j])
            # import pdb; pdb.set_trace()
            # dist_mat[i, j] = numpy.abs(
            #     data[frames1[i], 0] - data[frames2[j], 0])
    return dist_mat
