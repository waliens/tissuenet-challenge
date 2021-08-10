import csv
from collections import namedtuple
from tempfile import TemporaryDirectory

import cv2
import os
import time
from argparse import ArgumentParser

from cv2.cv2 import imread
from cytomine.cytomine import _cytomine_parameter_name_synonyms, Cytomine
from cytomine.models import Project, StorageCollection, ImageInstanceCollection, AnnotationCollection, Annotation, \
    Property
from cytomine.models.collection import CollectionPartialUploadException

from shapely.affinity import affine_transform
from shapely.geometry import box, Point, LineString, MultiPolygon, Polygon

import numpy as np

from xml.etree import ElementTree

METADATA = {
    "TCGA-A7-A13E-01Z-00-DX1": ("Breast", "Breast invasive carcinoma", "train"),
    "TCGA-A7-A13F-01Z-00-DX1": ("Breast", "Breast invasive carcinoma", "train"),
    "TCGA-AR-A1AK-01Z-00-DX1": ("Breast", "Breast invasive carcinoma", "train"),
    "TCGA-AR-A1AS-01Z-00-DX1": ("Breast", "Breast invasive carcinoma", "train"),
    "TCGA-E2-A1B5-01Z-00-DX1": ("Breast", "Breast invasive carcinoma", "train"),
    "TCGA-E2-A14V-01Z-00-DX1": ("Breast", "Breast invasive carcinoma", "train"),
    "TCGA-B0-5711-01Z-00-DX1": ("Kidney", "Kidney renal clear cell carcinoma", "train"),
    "TCGA-HE-7128-01Z-00-DX1": ("Kidney", "Kidney renal papillary cell carcinoma", "train"),
    "TCGA-HE-7129-01Z-00-DX1": ("Kidney", "Kidney renal papillary cell carcinoma", "train"),
    "TCGA-HE-7130-01Z-00-DX1": ("Kidney", "Kidney renal papillary cell carcinoma", "train"),
    "TCGA-B0-5710-01Z-00-DX1": ("Kidney", "Kidney renal clear cell carcinoma", "train"),
    "TCGA-B0-5698-01Z-00-DX1": ("Kidney", "Kidney renal clear cell carcinoma", "train"),
    "TCGA-18-5592-01Z-00-DX1": ("Liver", "Lung squamous cell carcinoma", "train"),
    "TCGA-38-6178-01Z-00-DX1": ("Liver", "Lung adenocarcinoma", "train"),
    "TCGA-49-4488-01Z-00-DX1": ("Liver", "Lung adenocarcinoma", "train"),
    "TCGA-50-5931-01Z-00-DX1": ("Liver", "Lung adenocarcinoma", "train"),
    "TCGA-21-5784-01Z-00-DX1": ("Liver", "Lung squamous cell carcinoma", "train"),
    "TCGA-21-5786-01Z-00-DX1": ("Liver", "Lung squamous cell carcinoma", "train"),
    "TCGA-G9-6336-01Z-00-DX1": ("Prostate", "Prostate adenocarcinoma", "train"),
    "TCGA-G9-6348-01Z-00-DX1": ("Prostate", "Prostate adenocarcinoma", "train"),
    "TCGA-G9-6356-01Z-00-DX1": ("Prostate", "Prostate adenocarcinoma", "train"),
    "TCGA-G9-6363-01Z-00-DX1": ("Prostate", "Prostate adenocarcinoma", "train"),
    "TCGA-CH-5767-01Z-00-DX1": ("Prostate", "Prostate adenocarcinoma", "train"),
    "TCGA-G9-6362-01Z-00-DX1": ("Prostate", "Prostate adenocarcinoma", "train"),
    "TCGA-DK-A2I6-01A-01-TS1": ("Bladder", "Bladder Urothelial Carcinoma", "train"),
    "TCGA-G2-A2EK-01A-02-TSB": ("Bladder", "Bladder Urothelial Carcinoma", "train"),
    "TCGA-AY-A8YK-01A-01-TS1": ("Colon", "Colon adenocarcinoma", "train"),
    "TCGA-NH-A8F7-01A-01-TS1": ("Colon", "Colon adenocarcinoma", "train"),
    "TCGA-KB-A93J-01A-01-TS1": ("Stomach", "Stomach adenocarcinoma", "train"),
    "TCGA-RD-A8N9-01A-01-TS1": ("Stomach", "Stomach adenocarcinoma", "train"),
    "TCGA-2Z-A9J9-01A-01-TS1": ("unknown", "unknown", "test"),
    "TCGA-44-2665-01B-06-BS6": ("unknown", "unknown", "test"),
    "TCGA-69-7764-01A-01-TS1": ("unknown", "unknown", "test"),
    "TCGA-A6-6782-01A-01-BS1": ("unknown", "unknown", "test"),
    "TCGA-AC-A2FO-01A-01-TS1": ("unknown", "unknown", "test"),
    "TCGA-AO-A0J2-01A-01-BSA": ("unknown", "unknown", "test"),
    "TCGA-CU-A0YN-01A-02-BSB": ("unknown", "unknown", "test"),
    "TCGA-EJ-A46H-01A-03-TSC": ("unknown", "unknown", "test"),
    "TCGA-FG-A4MU-01B-01-TS1": ("unknown", "unknown", "test"),
    "TCGA-GL-6846-01A-01-BS1": ("unknown", "unknown", "test"),
    "TCGA-HC-7209-01A-01-TS1": ("unknown", "unknown", "test"),
    "TCGA-HT-8564-01Z-00-DX1": ("unknown", "unknown", "test"),
    "TCGA-IZ-8196-01A-01-BS1": ("unknown", "unknown", "test"),
    "TCGA-ZF-A9R5-01A-01-TS1": ("unknown", "unknown", "test")
}


NUCLEI_TERM = 532821811


def linear_ring_is_valid(ring):
    points = set([(x, y) for x, y in ring.coords])
    return len(points) >= 3


def geom_as_list(geometry):
    """Return the list of sub-polygon a polygon is made up of"""
    if geometry.geom_type == "Polygon":
        return [geometry]
    elif geometry.geom_type == "MultiPolygon":
        return geometry.geoms


def fix_geometry(geometry):
    """Attempts to fix an invalid geometry (from https://goo.gl/nfivMh)"""
    if geometry.is_valid:
        return geometry
    try:
        return geometry.buffer(0)
    except ValueError:
        pass

    polygons = geom_as_list(geometry)

    fixed_polygons = list()
    for i, polygon in enumerate(polygons):
        if not linear_ring_is_valid(polygon.exterior):
            continue

        interiors = []
        for ring in polygon.interiors:
            if linear_ring_is_valid(ring):
                interiors.append(ring)

        fixed_polygon = Polygon(polygon.exterior, interiors)

        try:
            fixed_polygon = fixed_polygon.buffer(0)
        except ValueError:
            continue

        fixed_polygons.extend(geom_as_list(fixed_polygon))

    if len(fixed_polygons) > 0:
        return MultiPolygon(fixed_polygons)
    else:
        return None

def flatten_geoms(geoms):
    """Flatten (possibly nested) multipart geometry."""
    geometries = []
    for g in geoms:
        if hasattr(g, "geoms"):
            geometries.extend(flatten_geoms(g))
        else:
            geometries.append(g)
    return geometries



def get_polygons_from_xml(xmlfile_path, min_area=4):
    tree = ElementTree.parse(xmlfile_path)
    root = tree.getroot()

    polygons = list()
    for annotation in root[0].iter('Region'):
        # if annotation.attrib["Type"] != "Polygon":
        #     raise ValueError("not a polygon (but '{}') in {}".format(annotation.attrib["Type"], xmlfile_path))

        points = [(float(t.attrib["X"]), float(t.attrib["Y"])) for t in annotation.find("Vertices").findall("Vertex")]
        if len(points) == 0:
            continue
        elif len(points) == 1:
            polygon = Point(points[0])
        elif len(points) == 2:
            polygon = LineString(points)
        else:
            polygon = fix_geometry(Polygon(points))

        for geom in flatten_geoms([polygon]):
            if geom.area > min_area:
                polygons.append(geom)
    return polygons


def get_resolution(xmlfile_path):
    tree = ElementTree.parse(xmlfile_path)
    root = tree.getroot()
    resol = root.attrib.get("MicronsPerPixel")
    if resol is not None:
        return float(resol)
    else:
        return resol

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

        images = ImageInstanceCollection().fetch_with_filter("project", params.id_project)
        existing = {i.originalFilename for i in images}

        upload_params = {
            "upload_host": params.upload_host,
            "id_storage": my_storage.id
        }

        # read grades
        images_dir = os.path.join(params.dir, "Tissue Images")
        annots_dir = os.path.join(params.dir, "Annotations")

        for filename in os.listdir(images_dir):
            if filename in existing:
                print("Skipping '{}', already in project".format(filename))
                continue
            filepath = os.path.join(images_dir, filename)
            name = filename.split(".", 1)[0]
            annot_path = os.path.join(annots_dir, name + ".xml")

            _ = conn.upload_image(
                filename=filepath,
                id_project=params.id_project,
                **upload_params
            )

            image_instance = find_imageinstance(filename, id_project=params.id_project, wait_delay=1)
            resolution = get_resolution(annot_path)
            if resolution is not None:
                image_instance.physicalSizeX = resolution
                image_instance.physicalSizeY = resolution
                image_instance.save()
            Property(image_instance, key="organ", value=METADATA[name][0]).save()
            Property(image_instance, key="cancer", value=METADATA[name][1]).save()
            Property(image_instance, key="set", value=METADATA[name][2]).save()

            bbox = box(0, 0, image_instance.width, image_instance.height)

            if os.path.isfile(annot_path):
                objects = get_polygons_from_xml(annot_path)


                # for polygon in objects:
                #     annot = Annotation(
                #         location=change_referential(polygon.intersection(bbox), image_instance.height).wkt,
                #         id_terms=[NUCLEI_TERM],
                #         id_image=image_instance.id,
                #         id_project=params.id_project
                #     )
                #     annot.save()
                try:
                    collection = AnnotationCollection()
                    collection.extend([Annotation(
                            location=change_referential(polygon.intersection(bbox), image_instance.height).wkt,
                            id_terms=[NUCLEI_TERM],
                            id_image=image_instance.id,
                            id_project=params.id_project
                        ) for polygon in objects])
                    collection.save()
                except CollectionPartialUploadException as e:
                    print("Partial upload: '{}'".format(e))
                    continue


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
