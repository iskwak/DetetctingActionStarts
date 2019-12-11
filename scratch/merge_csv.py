"""Scratch file to merge all predictions into one csv."""
import os

source = "/media/drive1/kwaki/video_scratch/M147_20150302_v011"
# source = "/media/drive1/kwaki/video_scratch/M173_20150423_v033"
out_dir = "/media/drive1/kwaki/video_scratch/video"
out_predict = os.path.join(out_dir, "predict_all.csv")

names = [
    "predict_lift.csv",
    "predict_hand.csv",
    "predict_grab.csv",
    "predict_suppinate.csv",
    "predict_mouth.csv",
    "predict_chew.csv"
]

data = []
for name in names:
    full_name = os.path.join(source, name)
    print full_name

    lines = []
    with open(full_name, "r") as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    vals = []
    frames = []
    images = []
    for i in range(len(lines) - 1):
        temp = lines[i + 1]
        temp2 = temp.split(",")
        vals.append(temp2[1])

        frames.append(temp2[0])
        images.append(temp2[-1])

    data.append(vals)

# now write the data
with open(out_predict, "w") as f:
    f.write("frame,lift,hand,grab,suppinate,mouth,chew,image\n")

    # num_els = len(data[0])
    # for i in range(40, num_els):
    for i in range(180, 850):
        f.write("%s" % frames[i])
        for j in range(0, len(names)):
            f.write(",%s" % data[j][i])
        f.write(",%s" % images[i])
        f.write("\n")
