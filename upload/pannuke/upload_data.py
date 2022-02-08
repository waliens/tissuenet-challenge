

import os
import time
from argparse import ArgumentParser

import cv2
import numpy as np
import tempfile
from cytomine.cytomine import _cytomine_parameter_name_synonyms, Cytomine
from cytomine.models import ImageInstanceCollection, AnnotationCollection, Project, StorageCollection, Property, \
    Annotation
from cytomine.models.collection import CollectionPartialUploadException
from numpy.lib.format import open_memmap

from shapely.affinity import affine_transform
from sldc.locator import mask_to_objects_2d

TERMS = {
    0: (533204518, "Neoplastic cell"),
    1: (533204524, "Inflammatory"),
    2: (533204539, "Connective/Soft tissue cells"),
    3: (533204547, "Dead cell"),
    4: (533204555, "Epithelial"),
    5: (533204563, "Background"),
}


def change_referential(polygon, height):
    return affine_transform(polygon, [1, 0, 0, -1, 0, height])


def find_imageinstance(filename, id_project, max_attempt=10, wait_delay=1):
    attempt = 0
    while attempt < max_attempt:
        collection = ImageInstanceCollection().fetch_with_filter("project", id_project)

        filtered = list(filter(lambda i: i.originalFilename == filename, collection))
        if len(filtered) > 1:
            raise ValueError("Several file with the same name in the project")
        elif len(filtered) == 0:
            time.sleep(wait_delay)
        else:
            return filtered[0]

        attempt += 1
    return None


def main(argv):
    with Cytomine.connect_from_cli(argv) as conn:
        argparse = Cytomine._add_cytomine_cli_args(ArgumentParser())
        argparse.add_argument(*_cytomine_parameter_name_synonyms("id_project"),
                              dest="id_project", type=int, help="The Cytomine project id.", required=True)
        argparse.add_argument('--cytomine_upload_host', dest='upload_host',
                              default='research-upload.cytomine.be', help="The Cytomine upload host")
        argparse.add_argument('--dir', dest="dir")
        params, _ = argparse.parse_known_args(args=argv)

        # Check that the given project exists
        if params.id_project:
            project = Project().fetch(params.id_project)
            if not project:
                raise ValueError("Project not found")

        # To upload the image, we need to know the ID of your Cytomine storage.
        storages = StorageCollection().fetch()
        my_storage = next(filter(lambda storage: storage.user == conn.current_user.id, storages))
        if not my_storage:
            raise ValueError("Storage not found")

        init_images = ImageInstanceCollection().fetch_with_filter("project", params.id_project)
        existing = {i.originalFilename for i in init_images}

        upload_params = {
            "upload_host": params.upload_host,
            "id_storage": my_storage.id
        }

        images_dirname = os.path.join(params.dir, "images")
        masks_dirname = os.path.join(params.dir, "masks")
        for fold_nb in [1]: #, 2, 3]:
            fold = "fold{}".format(fold_nb)
            images_fold = os.path.join(images_dirname, fold)
            masks_fold = os.path.join(masks_dirname, fold)
            images = open_memmap(os.path.join(images_fold, "images.npy"))
            types = open_memmap(os.path.join(images_fold, "types.npy"))
            masks = open_memmap(os.path.join(masks_fold, "masks.npy"), dtype=np.uint8, shape=images.shape + (6,))
            n_images = images.shape[0]

            with tempfile.TemporaryDirectory() as tmpdir:

                collection = AnnotationCollection()
                for i in range(n_images):
                    img = images[i]
                    mask = masks[i]

                    filename = "pannuke_{}_{}.png".format(fold, i)
                    filepath = os.path.join(tmpdir, filename)

                    if filename not in existing:
                        cv2.imwrite(filepath, img)

                        _ = conn.upload_image(
                            filename=filepath,
                            id_project=params.id_project,
                            **upload_params
                        )

                        image_instance = find_imageinstance(filename, id_project=params.id_project, wait_delay=1)
                        Property(image_instance, key="fold", value=fold_nb).save()
                        Property(image_instance, key="organ", value=types[i]).save()
                        existing.add(filename)
                    else:
                        image_instance = init_images.find_by_attribute("originalFilename", filename)
                        if image_instance is None:
                            raise ValueError("image not found")

                        annots = AnnotationCollection(image=image_instance.id).fetch()

                        for a in annots:
                            a.delete()

                    for depth in range(5):
                        bmask = mask[:, :, depth].astype(np.uint8) * 255

                        if not np.any(bmask):
                            continue

                        bmask = cv2.morphologyEx(
                            src=bmask,
                            op=cv2.MORPH_ERODE,
                            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                        objects = [(o, l) for o, l in mask_to_objects_2d(bmask) if o.area > 4]

                        collection.extend([
                            Annotation(
                                location=change_referential(obj.buffer(1), image_instance.height).wkt,
                                term=TERMS[depth][0],
                                id_image=image_instance.id,
                                id_project=params.id_project)
                            for obj, _ in objects
                        ])

                try:
                    collection.save()
                except CollectionPartialUploadException as e:
                    print(e)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
