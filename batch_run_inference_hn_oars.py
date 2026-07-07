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

 
def _detect_input_scenario(input_path):
    """Detect input type (single/batch, DICOM/NIfTI) from directory contents.
 
    Args:
        input_path (str): Path to input directory.
 
    Returns:
            scenario - one of 'batch_dcm', 'batch_nii', 'single_dcm', 'single_nii'
            items    - list of paths to process (subdirs or files)
    """
    contents = [os.path.join(input_path, f) for f in sorted(os.listdir(input_path))]
    if not contents:
        raise ValueError(f"Input directory is empty: {input_path}")
 
    subdirs = [f for f in contents if os.path.isdir(f)]
    nii_files = [f for f in contents if _is_nii_file(f)]
    dcm_files = [f for f in contents if
                 os.path.isfile(f) and f.lower().endswith('.dcm')]
 
    # DICOM (batch)
    if subdirs and all(_is_dicom_dir(d) for d in subdirs) and not nii_files:
        return 'batch_dcm', subdirs
 
    # NIfTI files 
    if nii_files and not dcm_files and not subdirs:
        if len(nii_files) == 1:
            return 'single_nii', nii_files
        return 'batch_nii', nii_files
 
    # DICOM (single)
    if dcm_files and not nii_files and not subdirs:
        return 'single_dcm', [input_path]
 
    raise ValueError(
        f"Unrecognised input directory structure in: {input_path}.\n"
        f"Expected one of:\n"
        f"  1. Subdirectories each containing DICOM files (batch DICOM)\n"
        f"  2. Flat directory with multiple NIfTI files (batch NIfTI)\n"
        f"  3. Flat directory with DICOM files directly (single DICOM)\n"
        f"  4. Flat directory with a single NIfTI file (single NIfTI)"
    )
 



def main(inputPath, sessionPath, outputPath):

    # Create output dir
    os.makedirs(outputPath, exist_ok=True)

    # Examine input directory contents
    inputType, items = _detect_input_scenario(inputPath)
    contents = [os.path.join(inputPath, f) for f in os.listdir(inputPath)]

    # Run batch auto-seg
    total = len(items)
    for count, item in enumerate(items, 1):
        print(f"Segmenting item {count} of {total}: {item}")
        if inputType == 'batch_dcm':
            ptID = Path(item).stem
            ptSessionDir = os.path.join(sessionPath, ptID)
            run_inference_deeplab.main(item, ptSessionDir, outputPath,
                                       DCMexportFlag=True)
            run_inference_selfattn.main(item, ptSessionDir, outputPath,
                                        DCMexportFlag=True)
 
        elif inputType == 'single_dcm':
            ptID = Path(item).stem
            ptSessionDir = os.path.join(sessionPath, ptID)
            run_inference_deeplab.main(item, ptSessionDir, outputPath,
                                       DCMexportFlag=True)
            run_inference_selfattn.main(item, ptSessionDir, outputPath,
                                        DCMexportFlag=True)
 
        elif inputType in ('batch_nii', 'single_nii'):
            ptID = Path(item).stem.replace('.nii', '')
            ptSessionDir = os.path.join(sessionPath, ptID)
            run_inference_deeplab.main(item, ptSessionDir, outputPath,
                                       DCMexportFlag=False)
            run_inference_selfattn.main(item, ptSessionDir, outputPath,
                                        DCMexportFlag=False)
 

    return 0


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
