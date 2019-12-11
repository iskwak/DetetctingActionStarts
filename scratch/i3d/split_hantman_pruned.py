# split the movie hantman movie files.
import os
import cv2


def split_movie(exp, movie_dir, out_dir):
    movie_name = os.path.join(movie_dir, exp, "movie_comb.avi")
    side_name = os.path.join(out_dir, "side", "%s.avi" % exp)
    front_name = os.path.join(out_dir, "front", "%s.avi" % exp)
    # print(movie_name)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap = cv2.VideoCapture(movie_name)
    front_writer = cv2.VideoWriter(front_name, fourcc, 30.0, (352, 260))
    side_writer = cv2.VideoWriter(side_name, fourcc, 30.0, (352, 260))

    ret, frame = cap.read()
    while ret:
        front = frame[:, :352, :]
        side = frame[:, 352:, :]

        front_writer.write(front)
        side_writer.write(side)
        ret, frame = cap.read()

    cap.release()
    front_writer.release()
    side_writer.release()


def main():
    flist = "/nrs/branson/kwaki/data/lists/hantman_exp_list.txt"
    movie_dir = "/nrs/branson/kwaki/data/hantman_pruned"
    out_dir = "/nrs/branson/kwaki/data/videos/hantman_split/"
    with open(flist, "r") as fid:
        line = fid.readline()
        while line:
            # movie_name = os.path.join(
            #     movie_dir, line.strip(), "movie_comb.avi"
            # )
            exp = line.strip()
            print(exp)
            split_movie(exp, movie_dir, out_dir)
            line = fid.readline()


if __name__ == "__main__":
    main()
