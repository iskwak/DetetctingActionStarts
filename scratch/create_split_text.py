import os


def main():
    # base_list = "/nrs/branson/kwaki/data/lists/hantman_exp_list.txt"
    # num_vids = 1169


    base_list = "/groups/branson/bransonlab/kwaki/data/thumos14/lists/videos.txt"
    base_out = "/groups/branson/bransonlab/kwaki/data/thumos14/lists/split_32/"
    num_vids = 412
    num_splits = 32
    remainder = num_vids % num_splits
    base_num_vids = int(num_vids / num_splits)

    with open(base_list, "r") as list_fid:
        for i in range(num_splits):
            split_name = os.path.join(base_out, "list_%d.txt" % i)
            with open(split_name, "w") as split_fid:
                for j in range(base_num_vids):
                    exp_name = list_fid.readline()
                    exp_name = exp_name.strip()
                    # print(exp_name)
                    split_fid.write("%s\n" % exp_name)
                if i < remainder:
                    exp_name = list_fid.readline()
                    exp_name = exp_name.strip()
                    # print(exp_name)
                    split_fid.write("%s\n" % exp_name)


if __name__ == "__main__":
    main()
