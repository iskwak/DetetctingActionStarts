import os


def main():
    base_dir = '/nrs/branson/kwaki/data/hantman_flow'
    org_vids = os.path.join(base_dir, 'org_vids')

    vids = os.listdir(org_vids)
    for vid_name in vids:
        org_name = os.path.join(org_vids, vid_name)
        filename_parts = vid_name.split("_")
        new_name = "%s_%s_%s.avi" %\
            (filename_parts[0], filename_parts[1], filename_parts[2])

        if "side" in vid_name:
            new_name = os.path.join(base_dir, "side", new_name)
            os.symlink(org_name, new_name)
        elif "front" in vid_name:
            new_name = os.path.join(base_dir, "front", new_name)
            os.symlink(org_name, new_name)


if __name__ == "__main__":
    main()
