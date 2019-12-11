import os
from shutil import copyfile

def main():
    input_folder = "/nrs/branson/kwaki/data/hantman_flow/org_vids/"
    output_folder = "/nrs/branson/kwaki/data/videos/hantman_flow"

    vid_names = os.listdir(input_folder)
    vid_names.sort()

    for vid_name in vid_names:
        print(vid_name)
        src_name = os.path.join(input_folder, vid_name)

        temp = vid_name.split("_")
        if 'front' in temp[3]:
            view = 'front'
        else:
            view = 'side'

        dst_name = os.path.join(
            output_folder, view, "%s_%s_%s.avi" % (temp[0], temp[1], temp[2]))

        copyfile(src_name, dst_name)
        # print("\t%s,%s" % (src_name, dst_name))

if __name__ == "__main__":
    main()
