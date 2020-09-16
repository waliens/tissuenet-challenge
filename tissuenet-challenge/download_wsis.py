import csv
import os
import requests
from cytomine import Cytomine
from cytomine.models._utilities import generic_download


def split_s3_filename(filename):
    _, path = filename.split("://", 1)
    bucket, path = path.split("/", 1)
    return bucket, path


def format_size(size):
    if size > 10 ** 9:
        return "{:.2f}Gb".format(size / (10 ** 9))
    if size > 10 ** 6:
        return "{:.2f}Mb".format(size / (10 ** 6))
    if size > 10 ** 3:
        return "{:.2f}Kb".format(size / (10 ** 3))
    return "{}b".format(size)


def download_image(url, dst):
    if os.path.isfile(dst):
        print("image in cache '{}'".format(url))
        return
    else:
        print("download '{}'".format(url))
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dst, 'wb') as f:
            size = 0
            for chunk in r.iter_content(chunk_size=8192):
                size += len(chunk)
                # refresh += 1
                # if refresh == 50:
                #     print("\r{}".format(format_size(size)),  end="")
                #     sys.stdout.flush()
                #     refresh = 0
                f.write(chunk)
        print("done with '{}'".format(dst))


def download(args):
    url, path = args
    download_image(url, path)


def main(argv):
    with Cytomine.connect_from_cli(argv):
        download_path = argv[-2]
        n_jobs = int(argv[-1])
        os.makedirs(download_path, exist_ok=True)

        # with open("train_labels.csv") as file:
        #     reader = csv.DictReader(file, fieldnames=["filename", "0", "1", "2", "3"])
        #     next(reader)
        #     file2label = {line['filename']: [i - 1 for i, (k, v) in enumerate(line.items()) if k != "filename" and int(v) == 1][0] for line in reader}
        #
        # print(file2label)
        #
        # with open("train_annotations_lbzOVuS.csv") as file:
        #     reader = csv.DictReader(file, fieldnames=["annotation_id", "filename", "geometry", "annotation_class", "us_jpeg_url", "eu_jpeg_url", "asia_jpeg_url"])
        #     next(reader)
        #
        #     lesion_counts = defaultdict(lambda: defaultdict(lambda: 0))
        #
        #     urls = list()
        #     for line in reader:
        #         cls = int(line["annotation_class"])
        #         lesion_counts[line["filename"]][cls] += 1
        #         bucket, url_path = split_s3_filename(line['eu_jpeg_url'])
        #         dst_path = os.path.join(download_path, str(cls))
        #         os.makedirs(dst_path, exist_ok=True)
        #         fname, ext = line['filename'].rsplit(".", 1)
        #         filename = line['annotation_id'] + "-" + str(file2label[line['filename']]) + "." + ext
        #         urls.append(("https://" + bucket + ".s3.amazonaws.com/" + url_path, os.path.join(dst_path, filename)))
        #
        #     generic_download(urls, download, n_workers=4)
        #
        # print(len(file2label), len(lesion_counts))
        #
        # for filename, cls in sorted(file2label.items(), key=lambda v: v[1]):
        #     print(filename[:10], "[{}]".format(cls), " ".join([str(lesion_counts[filename].get(i, ".")) for i in range(4)]))

        fieldnames = [
            "filename", "width", "height", "resolution", "magnification", "tif_cksum", "tif_size", "us_wsi_url",
            "us_tif_url", "us_jpg_url", "eu_wsi_url", "eu_tif_url", "eu_jpg_url", "asia_wsi_url", "asia_tif_url",
            "asia_jpg_url"
        ]
        with open("train_metadata_eRORy1H.csv", "r") as file:
            reader = csv.DictReader(file, fieldnames=fieldnames)
            next(reader)

            urls = list()
            for line in reader:
                bucket, path = split_s3_filename(line['eu_tif_url'])
                os.makedirs(download_path, exist_ok=True)
                filepath = os.path.join(download_path, os.path.basename(line['eu_tif_url']))
                print(filepath)
                urls.append(("https://" + bucket + ".s3.amazonaws.com/" + path, filepath))

            generic_download(urls, download, n_workers=n_jobs)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])