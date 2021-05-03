import os
import csv
import numpy as np


def write_submission(preds):
    with open("submission.csv", "w+") as file:
        file.write("filename,0,1,2,3\n")
        for filename, pred_cls in preds.items():
            file.write(os.path.basename(filename) + "," + ",".join([str(int(pred_cls == cls)) for cls in range(4)]) + "\n")


def read_test_files():
    with open("data/test_metadata.csv", "r") as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)
        filenames = list()
        for row in reader:
            filenames.append(os.path.join("assets", row[0]))
        return filenames


def main(argv):
    test_files = read_test_files()
    np.random.seed(42)
    preds = {filename: np.random.randint(0, 4) for filename in test_files}
    write_submission(preds)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
