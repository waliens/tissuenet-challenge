import os
import sys
from pathlib import Path

from cytomine import Cytomine
from cytomine.models import ImageInstance, ImageInstanceCollection


def main(argv):
    """
    IMAGES VALID:
    * 005-TS_13C08351_2-2014-02-12 12.22.44.ndpi | id : 77150767
    * 024-12C07162_2A-2012-08-14-17.21.05.jp2 | id : 77150761
    * 019-CP_12C04234_2-2012-08-10-12.49.26.jp2 | id : 77150809

    IMAGES TEST:
    * 004-PF_08C11886_1-2012-08-09-19.05.53.jp2 | id : 77150623
    * 011-TS_13C10153_3-2014-02-13 15.22.21.ndpi | id : 77150611
    * 018-PF_07C18435_1-2012-08-17-00.55.09.jp2 | id : 77150755
    """
    with Cytomine.connect_from_cli(argv):
        path = os.path.join(str(Path.home()), "data", "thyroid", "wsi")
        os.makedirs(path, exist_ok=True)

        to_fetch = ImageInstanceCollection().fetch_with_filter("project", 77150529)
        for instance in to_fetch:
            instance.download(dest_pattern=os.path.join(path, "{originalFilename}"), override=False)


if __name__ == "__main__":
    main(sys.argv[1:])