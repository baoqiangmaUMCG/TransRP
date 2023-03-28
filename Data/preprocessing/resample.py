# The srcipt was created by Baoqiang to extract CT, PET and GTV in bounding box regions as well as resample them to 1x1x1 mm3 resolution.

from pathlib import Path
from multiprocessing import Pool
import logging

import click
import pandas as pd
import numpy as np
import SimpleITK as sitk

# Default paths
path_in = './Data/hecktor2022/imagesTr/'
path_label_in = './Data/hecktor2022/labelsTr/'
path_out = './Data/hecktor2022/resampled/'
path_bb = './Data/hecktor2022/bb_box/bb_box_training.csv'

@click.command()
@click.argument('input_folder', type=click.Path(exists=True), default=path_in)
@click.argument('input_label_folder', type=click.Path(exists=True), default=path_label_in)
@click.argument('output_folder', type=click.Path(), default=path_out)
@click.argument('bounding_boxes_file', type=click.Path(), default=path_bb)
@click.option('--cores',
              type=click.INT,
              default=12,
              help='The number of workers for parallelization.')
@click.option('--resampling',
              type=click.FLOAT,
              nargs=3,
              default=(1, 1, 1),
              help='Expect 3 positive floats describing the output '
              'resolution of the resampling. To avoid resampling '
              'on one or more dimension a value of -1 can be fed '
              'e.g. --resampling 1.0 1.0 -1 will resample the x '
              'and y axis at 1 mm/px and left the z axis untouched.')
def main(input_folder,input_label_folder, output_folder, bounding_boxes_file, cores, resampling):
    """ This command line interface allows to resample NIFTI files within a
        given bounding box contain in BOUNDING_BOXES_FILE. The images are
        resampled with spline interpolation
        of degree 3 and the segmentation are resampled
        by nearest neighbor interpolation.

        INPUT_FOLDER is the path of the folder containing the NIFTI to
        resample.
        OUTPUT_FOLDER is the path of the folder where to store the
        resampled NIFTI files.
        BOUNDING_BOXES_FILE is the path of the .csv file containing the
        bounding boxes of each patient.
    """
    logger = logging.getLogger(__name__)
    logger.info('Resampling')

    input_folder = Path(input_folder)
    input_label_folder = Path(input_label_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    print('resampling is {}'.format(str(resampling)))
    bb_df = pd.read_csv(bounding_boxes_file)
    bb_df = bb_df.set_index('PatientID')
    
    #patient_list = [f.name.split("_")[0] for f in input_folder.rglob("*__CT*")]
    patient_list = list(pd.read_csv(path_bb)['PatientID'])
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(resampling)

    def resample_one_patient(p):
        bb = np.array([
            bb_df.loc[p, 'x1'] - 24, bb_df.loc[p, 'y1'] - 12, bb_df.loc[p, 'z1'] - 48,
            bb_df.loc[p, 'x2'] + 24, bb_df.loc[p, 'y2'] + 36, bb_df.loc[p, 'z2']
        ])
        size = np.round((bb[3:] - bb[:3]) / resampling).astype(int)
        ct = sitk.ReadImage(
            str([f for f in input_folder.rglob(p + "__CT*")][0].resolve()))
        pt = sitk.ReadImage(
            str([f for f in input_folder.rglob(p + "__PT*")][0].resolve()))
        gtvt = sitk.ReadImage(
            str([f for f in input_label_folder.rglob(p + "*")][0].resolve()))
        resampler.SetOutputOrigin(bb[:3])
        resampler.SetSize([int(k) for k in size])  # sitk is so stupid
        resampler.SetInterpolator(sitk.sitkBSpline)
        ct = resampler.Execute(ct)
        pt = resampler.Execute(pt)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        gtvt = resampler.Execute(gtvt)
        sitk.WriteImage(ct, str(
            (output_folder / (p + "__CT.nii.gz")).resolve()))
        sitk.WriteImage(pt, str(
            (output_folder / (p + "__PT.nii.gz")).resolve()))
        sitk.WriteImage(gtvt,
                        str((output_folder / (p + "__gtv.nii.gz")).resolve()))

    for p in patient_list:
    
        resample_one_patient(p)
    # with Pool(cores) as p:
    #    p.map(resample_one_patient, patient_list)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()
