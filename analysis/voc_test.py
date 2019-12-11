"""Test out the voc_test code."""
import numpy as np

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
            import pdb; pdb.set_trace()
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        import pdb; pdb.set_trace()
    return ap


def voc_ap2(rec, tps, fps, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    # 11 point metric
    ap = 0.
    fp = np.cumsum(fps)
    tp = np.cumsum(tps)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    aps = []
    for t in np.arange(0.1, 1.1, 0.1):
        p = 0
        # loop backwards until first tps == 1
        idx = np.argwhere(rec <= t).flatten()
        p = 0
        rel = 0
        for i in idx:
            p = p + prec[i] * fps[i]
            rel = rel + tps[i]
        aps.append(p / rel)
    # aps = prec * tps / sum(tps)
    import pdb; pdb.set_trace()

    return ap


def voc_eval(classname,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # # extract gt objects for this class
    # class_recs = {}
    # npos = 0
    # for imagename in imagenames:
    #     import pdb; pdb.set_trace()
    #     R = [obj for obj in recs[imagename] if obj['name'] == classname]
    #     bbox = np.array([x['bbox'] for x in R])
    #     difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    #     det = [False] * len(R)
    #     npos = npos + sum(~difficult)
    #     class_recs[imagename] = {'bbox': bbox,
    #                              'difficult': difficult,
    #                              'det': det}

    # # read dets
    # detfile = detpath.format(classname)
    # with open(detfile, 'r') as f:
    #     lines = f.readlines()

    # splitlines = [x.strip().split(' ') for x in lines]
    # image_ids = [x[0] for x in splitlines]
    # confidence = np.array([float(x[1]) for x in splitlines])
    # BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # # sort by confidence
    # sorted_ind = np.argsort(-confidence)
    # sorted_scores = np.sort(-confidence)
    # BB = BB[sorted_ind, :]
    # image_ids = [image_ids[x] for x in sorted_ind]

    # # go down dets and mark TPs and FPs
    # nd = len(image_ids)
    # tp = np.zeros(nd)
    # fp = np.zeros(nd)
    # for d in range(nd):
    #     R = class_recs[image_ids[d]]
    #     bb = BB[d, :].astype(float)
    #     ovmax = -np.inf
    #     BBGT = R['bbox'].astype(float)

    #     if BBGT.size > 0:
    #         # compute overlaps
    #         # intersection
    #         ixmin = np.maximum(BBGT[:, 0], bb[0])
    #         iymin = np.maximum(BBGT[:, 1], bb[1])
    #         ixmax = np.minimum(BBGT[:, 2], bb[2])
    #         iymax = np.minimum(BBGT[:, 3], bb[3])
    #         iw = np.maximum(ixmax - ixmin + 1., 0.)
    #         ih = np.maximum(iymax - iymin + 1., 0.)
    #         inters = iw * ih

    #         # union
    #         uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
    #                (BBGT[:, 2] - BBGT[:, 0] + 1.) *
    #                (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

    #         overlaps = inters / uni
    #         ovmax = np.max(overlaps)
    #         jmax = np.argmax(overlaps)

    #     if ovmax > ovthresh:
    #         if not R['difficult'][jmax]:
    #             if not R['det'][jmax]:
    #                 tp[d] = 1.
    #                 R['det'][jmax] = 1
    #             else:
    #                 fp[d] = 1.
    #     else:
    #         fp[d] = 1.
    npos = 8
    tps = [0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    fps = [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]

    # compute precision recall
    fp = np.cumsum(fps)
    tp = np.cumsum(tps)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # ap = voc_ap(rec, prec, use_07_metric=False)
    ap = voc_ap2(rec, tps, fps, True)
    import pdb; pdb.set_trace()

    return rec, prec, ap


if __name__ == "__main__":
    voc_eval("moo")
