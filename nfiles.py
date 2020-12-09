import argparse
import os
import sys


def countFiles(folder):
    files = os.listdir(folder)
    files = [os.path.join(folder, f) for f in files]
    files = [f for f in files if os.path.isfile(f)]
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    root_dir = args.root_dir
    output = args.output

    total = 0
    labellist = os.listdir(root_dir)
    labelpaths = [os.path.join(root_dir, l) for l in labellist]
    filename = "nfiles_" + root_dir + ".txt"

    countlist = []
    for label in labellist:
        labelpath = os.path.join(root_dir, label)
        count = countFiles(labelpath)
        total += count
        counttuple = (label, count)
        countlist.append(counttuple)
    countlist = sorted(countlist, reverse=True, key=lambda x: x[1])
    print("total number of files: %d" % total)

    if output:
        with open(output, "w") as f:
            for pair in countlist:
                f.write("{}, {}\n".format(pair[0], pair[1]))
