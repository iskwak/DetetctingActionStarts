"""Helper script to link the C3D features to the base hdf5 files."""
import h5py
import os
import numpy


def get_exp_names(all_exps, h5_filename):
    """Helper function to get the exp names from the hdf5 file."""
    with h5py.File(h5_filename, "r") as h5_data:
        exp_names = h5_data["exp_names"][()].astype("U")
        all_exps.append(exp_names)

def main():
    """Create external links to C3D hdf5 files."""
    # loop over each of the hdf5 files and create a global list experiments
    # to explore... Needed elsewhere anyways.
    base_dir = "/nrs/branson/kwaki/data/20180729_base_hantman"
    exp_dir = os.path.join(base_dir, "exps")
    list_dir = "/nrs/branson/kwaki/data/lists/"

    # see if the file list exists
    list_file = os.path.join(list_dir, "hantman_exp_list.txt")
    # read the exp names.
    all_exps = []
    with open(list_file, "r") as fid:
        line = fid.readline()
        while line:
            all_exps.append(line.strip())
            line = fid.readline()

    all_exps = numpy.array(all_exps)
    # after getting the list, go through and create external links
    for exp_name in all_exps:
        print(exp_name)
        full_name = os.path.join(exp_dir, exp_name)
        with h5py.File(full_name, "a") as h5_data:
            h5_data["M173_finetune_i3d_flow_front"] = h5py.ExternalLink(
                os.path.join("M173","flow", "front", exp_name),
                "/finetune_i3d_flow_front"
            )
            h5_data["M173_finetune_i3d_flow_side"] = h5py.ExternalLink(
                os.path.join("M173","flow", "side", exp_name),
                "/finetune_i3d_flow_side"
            )
            h5_data["M173_finetune_i3d_rgb_front"] = h5py.ExternalLink(
                os.path.join("M173","rgb", "front", exp_name),
                "/finetune_i3d_rgb_front"
            )
            h5_data["M173_finetune_i3d_rgb_side"] = h5py.ExternalLink(
                os.path.join("M173","rgb", "side", exp_name),
                "/finetune_i3d_rgb_side"
            )

            # h5_data["canned_i3d_flow_front"] = h5py.ExternalLink(
            #     os.path.join("canned_i3d_flow", exp_name), "/canned_i3d_flow_front"
            # )
            # h5_data["canned_i3d_flow_side"] = h5py.ExternalLink(
            #     os.path.join("canned_i3d_flow", exp_name), "/canned_i3d_flow_side"
            # )
    print("hi")


if __name__ == "__main__":
    main()


# h5_data["rgb_i3d_view1_fc"] = h5py.ExternalLink(
#     os.path.join("i3d", exp_name), "/rgb_i3d_view1_fc"
# )
# h5_data["rgb_i3d_view2_fc"] = h5py.ExternalLink(
#     os.path.join("i3d", exp_name), "/rgb_i3d_view2_fc"
# )
# h5_data["rgb_i3d_view1_fc_64"] = h5py.ExternalLink(
#     os.path.join("i3d_64", exp_name), "/rgb_i3d_view1_fc"
# )
# h5_data["rgb_i3d_view2_fc_64"] = h5py.ExternalLink(
#     os.path.join("i3d_64", exp_name), "/rgb_i3d_view2_fc"
# )
# h5_data["flow_i3d_view1_fc"] = h5py.ExternalLink(
#     os.path.join("flow_i3d", exp_name), "/flow_i3d_view1_fc"
# )
# h5_data["flow_i3d_view2_fc"] = h5py.ExternalLink(
#     os.path.join("flow_i3d", exp_name), "/flow_i3d_view2_fc"
# )
# h5_data["pca_i3d"] = h5py.ExternalLink(
#     os.path.join("pca_i3d", exp_name), "/reduced_i3d"
# )
# h5_data["pca_i3d2"] = h5py.ExternalLink(
#     os.path.join("pca_i3d2", exp_name), "/pca_i3d2"
# )
