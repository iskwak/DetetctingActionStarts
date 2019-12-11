""""Wrapper for helpers.create_hantman_splits"""
import os
import helpers.create_hantman_splits as create_hantman_splits

all_test = [
    ("M134", "20150427"), ("M134", "20150325"), ("M134", "20150506"),
    ("M134", "20150505"), ("M147", "20150506"), ("M147", "20150427"),
    ("M147", "20141209"), ("M147", "20150302"), ("M173", "20150430"),
    ("M173", "20150512"), ("M173", "20150506"), ("M173", "20150505"),
    ("M174", "20150413"), ("M174", "20150417"), ("M174", "20150414"),
    ("M174", "20150416")
]


for test_set in all_test:
    test_mouse = test_set[0]
    test_date = test_set[1]
    base_out = "hantman_%s_%s" % (test_mouse, test_date)

    data_dir = "/nrs/branson/kwaki/data/20180729_base_hantman"
    data_file = os.path.join(data_dir, "data.hdf5")

    arg_string = [
        "helpers/create_hantman_splits.py",
        "--data", data_file,
        "--name", base_out,
        "--test_mouse", test_mouse,
        "--test_date", test_date,
        "--split_type", "3",
        "--prune"
    ]
    # "--data", "/nrs/branson/kwaki/data/20180605_base_hantman/data.hdf5",
    print(base_out)
    opts = create_hantman_splits._setup_opts(arg_string)
    create_hantman_splits.create_train_test(opts)



# # all mice:
# all_mice = ["M134", "M147", "M173", "M174"]
# # all_mice = ["M174"]

# for test_mouse in all_mice:
#     # need to loop over all mice and create train/test splits.
#     # test_mouse = "M134"
#     base_out = "hantman_split_%s" % test_mouse
#     data_dir = "/nrs/branson/kwaki/data/20180729_base_hantman"
#     data_file = os.path.join(data_dir, "data.hdf5")

#     arg_string = [
#         "helpers/create_hantman_splits.py",
#         "--data", data_file,
#         "--name", base_out,
#         "--test_mouse", test_mouse,
#         "--prune"
#     ]
#     # "--data", "/nrs/branson/kwaki/data/20180605_base_hantman/data.hdf5",
#     print(base_out)
#     opts = create_hantman_splits._setup_opts(arg_string)
#     create_hantman_splits.create_train_test(opts)

# # check the split data.
# print("checking splits.")
# import helpers.check_hantman_splits as check_hantman_splits

# for test_mouse_date in all_test:
#     test_mouse = test_mouse_date[0]
#     test_date = test_mouse_date[1]
#     base_out = "hantman_split_%s" % test_mouse

#     check_hantman_splits.main(
#         data_dir,
#         base_out
#     )
