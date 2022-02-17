
import os
import re
import time

import numpy as np

from argparse import ArgumentParser

from skimage.io import imread

from shapely.affinity import affine_transform
from sldc.locator import mask_to_objects_2d
from cytomine.cytomine import _cytomine_parameter_name_synonyms, Cytomine
from cytomine.models import Project, StorageCollection, ImageInstanceCollection, AnnotationCollection, Annotation, Property



TERMS = {
    "cytoplasm": 543560059,  # 20
    "nucleus": 543560090  # 40
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


def readall(file):
    with open(file, "r") as f:
        return [a.strip() for a in f.readlines()]


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

        images_at_start = ImageInstanceCollection().fetch_with_filter("project", params.id_project)
        existing_images = {image.originalFilename for image in images_at_start}

        upload_params = {
            "upload_host": params.upload_host,
            "id_storage": my_storage.id
        }

        for folder in ["train", "test", "validation"]:
            x_folder = os.path.join(params.dir, folder, "x")
            y_folder = os.path.join(params.dir, folder, "y")
            x_files = {int(x.rsplit(".", 1)[0][-4:]): os.path.join(x_folder, x) for x in os.listdir(x_folder) if x.endswith(".bmp")}
            if not os.path.exists(y_folder):
                continue
            y_files = os.listdir(y_folder)

            for index, x_filepath in x_files.items():
                x_filename = os.path.basename(x_filepath)
                pattern = re.compile(r"^"+str(index)+"_[0-9]+.bmp$")
                y_files_for_index = [filename for filename in y_files if pattern.match(filename) is not None]

                if x_filename not in existing_images:
                    _ = conn.upload_image(
                        filename=x_filepath,
                        id_project=params.id_project,
                        **upload_params
                    )

                image_instance = find_imageinstance(x_filename, id_project=params.id_project, wait_delay=1)

                if image_instance is None:
                    raise ValueError("could not find uploaded file")

                if x_filename not in existing_images:
                    Property(image_instance, key="set", value=folder).save()

                # delete any existing annotation
                old_annotations = AnnotationCollection(project=params.id_project, image=image_instance.id).fetch()

                to_upload = AnnotationCollection()

                for y_file in y_files_for_index:
                    mask = imread(os.path.join(y_folder, y_file))
                    if mask.ndim > 2:
                        if np.any(np.sum(mask, axis=2) != mask.shape[2] * mask[:, :, 0]):
                            raise ValueError("multi-channel mask image '{}' ({}) with different channel content".format(os.path.join(y_folder, y_file), mask.shape))
                        mask = mask[:, :, 0]
                    objects = mask_to_objects_2d(mask)

                    for polygon, label in objects:
                        if polygon.area < 20:
                            continue
                        to_upload.append(Annotation(
                            location=change_referential(polygon, image_instance.height).wkt,
                            id_image=image_instance.id,
                            id_terms=[TERMS['cytoplasm'] if label == 20 else TERMS['nucleus']],
                            id_project=params.id_project
                        ))

                if len(to_upload) != len(old_annotations):
                    for annot in old_annotations:
                        annot.delete()
                    to_upload.save()

        # for image_filename in files:
        #     image_name = str(image_filename.rsplit(".", 1)[0])
        #     image = images_meta[image_name]
        #
        #     _ = conn.upload_image(
        #         filename=os.path.join(params.dir, image_filename),
        #         id_project=params.id_project,
        #         **upload_params
        #     )
        #
        #     image_instance = find_imageinstance(image_filename, id_project=params.id_project, wait_delay=1)
        #     Property(image_instance, key="set", value=image.set).save()
        #     Property(image_instance, key="patient", value=image.patient).save()
        #
        #     # create polygons
        #     mask_filename = image_name + "_anno.bmp"
        #     mask_filepath = os.path.join(params.dir, mask_filename)
        #     mask = imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        #     objects = mask_to_objects_2d(mask)
        #
        #     collection = AnnotationCollection()
        #     for object in objects:
        #         collection.append(Annotation(
        #             location=change_referential(object.polygon, image_instance.height).wkt,
        #             id_image=image_instance.id,
        #             id_terms=[TERM_GLAND],
        #             id_project=params.id_project
        #         ))
        #
        #     collection.append(Annotation(
        #         location=box(0, 0, image_instance.width, image_instance.height).wkt,
        #         id_image=image_instance.id,
        #         id_terms=[image.level],
        #         id_project=params.id_project
        #     ))
        #     collection.save()



if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
