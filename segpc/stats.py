from collections import defaultdict

from PIL import ImageDraw
from PIL import Image
from cytomine import Cytomine
from cytomine.models import AnnotationCollection, Project, ImageInstanceCollection, PropertyCollection

import numpy as np
from shapely import wkt
from shapely.geometry.base import BaseMultipartGeometry
from skimage import morphology
from sldc.locator import mask_to_objects_2d


def has_alpha_channel(image):
    """Check whether the image has an alpha channel

    Parameters
    ----------
    image: ndarray
        The numpy representation of the image

    Returns
    -------
    has_alpha: boolean
        True if the image has an alpha channel, false otherwise
    """
    chan = image.shape
    return len(chan) == 3 and (chan[2] == 2 or chan[2] == 4)




def alpha_rasterize(image, polygon):
    """
    Rasterize the given polygon as an alpha mask of the given image. The
    polygon is assumed to be referenced to the top left pixel of the image.
    If the image has already an alpha mask it is replaced by the polygon mask

    Parameters
    ----------
    image: ndarray
        The numpy representation of the image
    polygon : Polygon
        The polygon to rasterize

    Return
    ------
    rasterized : ndarray
        The image (in numpy format) of the rasterization of the polygon.
        The image should have the same dimension as the bounding box of
        the polygon.
    """
    # destination image
    source = np.asarray(image)

    # extract width, height and number of channels
    chan = source.shape
    if len(chan) == 3:
        height, width, depth = source.shape
    else:
        height, width = source.shape
        depth = 1
        source = source.reshape((height, width, depth))

    # if there is already an alpha mask, replace it
    if has_alpha_channel(image):
        source = source[:, :, 0:depth-1]
    else:
        depth += 1

    # create rasterization mask
    if polygon.is_empty: # handle case when polygon is empty
        alpha = np.zeros((height, width), dtype=np.uint8)
    else:
        alpha = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(alpha)
        if isinstance(polygon, BaseMultipartGeometry):  # geometry collection
            for geometry in polygon.geoms:
                try:
                    # if the geometry has not boundary (for MultiPoint for instance), a value error is raised.
                    # In this case, just skip the drawing of the geometry
                    seq_pts = geometry.boundary.coords
                    draw.polygon(seq_pts, outline=0, fill=255)
                except NotImplementedError:
                    pass
        else:
            boundary = polygon.boundary
            if isinstance(boundary, BaseMultipartGeometry):
                for sub_boundary in boundary.geoms:
                    seq_pts = sub_boundary.coords
                    draw.polygon(seq_pts, outline=0, fill=255)
            else:
                seq_pts = boundary.coords
                draw.polygon(seq_pts, outline=0, fill=255)

    # merge mask with tensors
    rasterized = np.zeros((height, width, depth), dtype=source.dtype)
    rasterized[:, :, 0:depth-1] = source
    rasterized[:, :, depth-1] = alpha
    return rasterized


def group_by(data, key_fn):
    groups = defaultdict(list)
    for item in data:
        item_key = key_fn(item)
        groups[item_key].append(item)
    return groups


def get_set(image):
    properties = PropertyCollection(image).fetch()
    dict_props = properties.as_dict()
    return dict_props["set"].value if "set" in dict_props else "other"


def main(argv):
    with Cytomine.connect_from_cli(argv) as conn:
        project = Project().fetch(543560004)
        annotations = AnnotationCollection(project=project.id, showWKT=True).fetch()
        images = ImageInstanceCollection().fetch_with_filter("project", project.id)

        image_by_id = group_by(images, lambda i: i.id)
        image_by_set = group_by(images, get_set)
        annotations_by_image = group_by(annotations, lambda a: a.image)
        neighbourhood = morphology.ball(2)

        new_annotations_by_image = dict()

        for image_id, annotations in annotations_by_image.items():
            image = image_by_id[image_id][0]
            mask = np.zeros([image.height, image.width], dtype=np.uint8)
            for annot in annotations:
                mask = alpha_rasterize(mask, wkt.loads(annot.location))
            mask = morphology.closing(mask, neighbourhood)
            mask = morphology.opening(mask, neighbourhood)

            objects = mask_to_objects_2d(mask[:,:,0])
            new_annotations_by_image[image_id] = [p for p, _ in objects]

        annotations_by_image = new_annotations_by_image

        by_count = group_by(annotations_by_image.values(), lambda l: len(l))

        for s, imgs in image_by_set.items():
            print(s, len(imgs))

        for count, images in sorted(by_count.items(), key=lambda v: v[0]):
            print(count, len(images))

        print(len(annotations))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])