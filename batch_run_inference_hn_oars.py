import sys
import os
from pathlib import Path

from model_wrapper import run_inference_deeplab, run_inference_selfattn

def _is_dicom_dir(path):
    """Check if a directory contains DICOM files (.dcm)."""
    if not os.path.isdir(path):
        return False
    return any(f.lower().endswith('.dcm') for f in os.listdir(path))


def _is_nii_file(path):
    """Check if a file is a NIfTI file (.nii or .nii.gz)."""
    return os.path.isfile(path) and \
           (path.endswith('.nii') or path.endswith('.nii.gz'))


def main(inputPath, sessionPath, outputPath):

    # Create output dir
    os.makedirs(outputPath, exist_ok=True)

    # Examine input directory contents
    contents = [os.path.join(inputPath, f) for f in os.listdir(inputPath)]
    if not contents:
        print("Input directory is empty.")
        return

    # Identify input type
    dcmFlag = all(os.path.isdir(f) and _is_dicom_dir(f) for f in contents)
    niiFlag = all(_is_nii_file(f) for f in contents)

    if not dcm_flag and not nii_flag:
        raise ValueError(
            "Input directory must contain either subdirectories of DICOM files "
            "or a flat list of NIfTI (.nii / .nii.gz) files."
        )

    # Run batch auto-seg
    if dcmFlag:
        # Loop over pts
        count = 0
        for ptDir in contents:
            count = count + 1
            print('Segmenting dataset {} of {}'.format(count, len(contents)))
            ptID = Path(Path(ptDir).stem).stem
            ptSessionDir = os.path.join(sessionPath, ptID)
            run_inference_deeplab.main(ptDir, ptSessionDir, outputPath, DCMexportFlag=True)
            run_inference_selfattn.main(ptDir, ptSessionDir, outputPath, DCMexportFlag=True)

    elif niiFlag:
        fileList = os.listdir(inputPath)
        count = 0
        for fileName in fileList:
            count = count + 1
            print('Segmenting file {} of {}'.format(count, len(fileList)))
            ptID = os.path.basename(inputPath)
            filePath = os.path.join(inputPath, fileName)
            ptSessionDir = os.path.join(sessionPath, ptID)
            run_inference_deeplab.main(filePath, ptSessionDir, outputPath, DCMexportFlag=False)
            run_inference_selfattn.main(filePath, ptSessionDir, outputPath, DCMexportFlag=False)

    return 0


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
