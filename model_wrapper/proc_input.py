import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path

from cerr import plan_container as pc

IMG_EXTS = ('.nii.gz', '.nii', '.nrrd', '.nhdr')
MAX_LABELS = 40 #Images with fewer unique values than this are identified as masks

def _get_img_ext(path):
    base = os.path.basename(os.path.normpath(path))
    for ext in IMG_EXTS:
        if base.lower().endswith(ext):
            return base[:-len(ext)]
    return base

def _is_dicom_dir(path):
    """Check if a directory contains DICOM files (.dcm)."""
    if not os.path.isdir(path):
        return False
    return any(f.lower().endswith('.dcm') for f in os.listdir(path))

def _is_nii_dir(path):
    return os.path.isdir(path) and \
           any(_is_nii_file(os.path.join(path, f)) for f in os.listdir(path))

def _is_nii_file(path):
    """Check if a file is a NIfTI file."""
    return os.path.isfile(path) and \
           path.lower().endswith(IMG_EXTS)

def _is_nii_mask(path):
    arr = sitk.getArrayViewFromImage(sitk.ReadImage(path))
    return len(np.unique(arr)) <= MAX_LABELS

def _load_nii_data(scan_file, struct_files):
    planC = pc.loadNiiScan(scan_file, imageType="CT SCAN")
    for struct_file in struct_files:
        name = _get_img_ext(struct_file)
        is_binary = len(np.unique(sitk.GetArrayViewFromImage(sitk.ReadImage(struct_file)))) <= 2
        labels = {name: 1} if is_binary else {}
        planC = pc.loadNiiStructure(struct_file, 0, planC, labels)
    return planC

def _identify_scan_and_labels(dir_path):
    files = sorted(os.path.join(dir_path, f) for f in os.listdir(dir_path)
                   if _is_nii_file(os.path.join(dir_path, f)))
    scans = [f for f in files if not _is_nii_mask(f)]
    structs = [f for f in files if f not in scans]
    if len(scans)!=1:
        raise ValueError(f"Expected a single scan file in {dir_path}. Found {len(scans)}")
    return scans[0], structs

def load_input(input_path):
    if _is_nii_file(input_path):
        scan_file, struct_files = input_path, []
    elif _is_nii_dir(input_path):
        scan_file, struct_files = _identify_scan_and_labels(input_path)
    elif _is_dicom_dir(input_path):
        pt_id = Path(Path(input_path).stem).stem
        planC = pc.loadDcmDir(input_path)
        orig_img = None
        is_dcm = True
        return planC, pt_id, orig_img, is_dcm
    else:
        raise ValueError('Invalid input path ', input_path)

    planC = pc.loadNiiScan(scan_file, imageType="CT SCAN")
    for struct_file in struct_files:
        name = _get_img_ext(struct_file)
        is_binary = len(np.unique(sitk.GetArrayViewFromImage(
            sitk.ReadImage(struct_file)))) <= 2
        labels = {name: 1} if is_binary else {}   # always pass a fresh dict
        planC = pc.loadNiiStructure(struct_file, 0, planC, labels)

    return planC, _get_img_ext(input_path), sitk.ReadImage(scan_file), False