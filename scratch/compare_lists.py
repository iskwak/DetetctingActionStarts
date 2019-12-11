# make sure the lists makes sense.

file1 = "/nrs/branson/kwaki/data/lists/hantman_list.txt"
file2 = "/nrs/branson/kwaki/data/lists/hantman_list2.txt"

# get the files from each list and compare
flist1 = []
with open(file1, "r") as f1:
    for line in f1:
        flist1.append(line)

flist2 = []
with open(file2, "r") as f2:
    for line in f2:
        flist2.append(line)

flist1.sort()
flist2.sort()

print(set(flist1) - set(flist2))
