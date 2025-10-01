import fnmatch
import glob
import os
import sys
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch.utils.data

from cerr import plan_container as pc
from cerr.dataclasses import scan as cerrScn
from cerr.dataclasses import structure
from cerr.dcm_export import rtstruct_iod
from cerr.radiomics.preprocess import getResampledGrid, imgResample3D
from cerr.utils.ai_pipeline import getScanNumFromIdentifier
from cerr.utils.image_proc import resizeScanAndMask
from cerr.utils.mask import computeBoundingBox, getPatientOutline

from models.models import create_model
from options.train_options import TrainOptions

# Input image dimensions
input_size = 256

# Output structure name to label map
num_labels = 11
label_dict = {1:"Left Parotid", 2:"Right Parotid",
               3:"Left Submandible", 4:"Right Submandible",
               7:"Mandible", 8:"Spinal cord",
               9:"Brain stem", 10:"Oral cavity"}
output_str_labels = list(label_dict.keys())
output_str_names = list(label_dict.values())

model_arch = 'self_attn'

def find(pattern, path):
    """look for NIfTI files in the given directory"""
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def load_file(file_name):
    """ Read NIfTI image """
    input_img = sitk.ReadImage(file_name)
    scan = np.array(sitk.GetArrayFromImage(input_img))
    scan = np.moveaxis(scan, 0, -1)
    return scan, input_img


def process_image(planC):
    """Image pre-processing using pyCERR"""

    # Processing parameters
    modality = 'CT'
    identifier = {'imageType': 'CT SCAN'}

    grid_type = 'center'
    resamp_method = 'sitkBSpline'
    output_res = [0.1, 0.1, 0]  # Output res: 1mm x 1mm in-plane

    resize_method = 'pad2d'
    out_size = [256, 256]
    input_mask_arr = None

    outline_struct_name = 'outline'
    intensity_threshold = -400  # Air intensity for outline detection

    # Get scan array
    scan_num = getScanNumFromIdentifier(identifier, planC)[0]
    x_vals, y_vals, z_vals = planC.scan[scan_num].getScanXYZVals()
    scan_arr = planC.scan[scan_num].getScanArray()

    # 1. Resample
    [x_resample, y_resample, z_resample] = getResampledGrid(output_res,
                                                            x_vals, y_vals, z_vals,
                                                            grid_type)
    resamp_scan_arr = imgResample3D(scan_arr,
                                     x_vals, y_vals, z_vals,
                                     x_resample, y_resample, z_resample,
                                     resamp_method, inPlane=True)

    resample_grid = [x_resample, y_resample, z_resample]
    planC = pc.importScanArray(resamp_scan_arr,
                                   resample_grid[0], resample_grid[1], resample_grid[2],
                                   modality, scan_num, planC)
    resample_scan_num = len(planC.scan) - 1

    # 2. Extract patient outline
    replace_str_num = None
    outline_mask_arr = getPatientOutline(resamp_scan_arr, intensity_threshold)
    planC = pc.importStructureMask(outline_mask_arr, scan_num,
                                       outline_struct_name,
                                       planC, replace_str_num)

    # 3. Crop to patient outline on each slice
    sum_slices = np.sum(outline_mask_arr, axis=(0, 1))
    valid_slices = np.where(sum_slices > 0)[0]
    num_slices = len(valid_slices)
    limits = np.zeros((num_slices, 4))

    for slc in range(num_slices):
        minr, maxr, minc, maxc, _, _, _ = computeBoundingBox( \
                outline_mask_arr[:, :, valid_slices[slc]],
                is2DFlag=True)
        limits[slc, :] = [minr, maxr, minc, maxc]

    # 4. Resize to 256 x 256 in-plane
    resamp_slc_arr = resamp_scan_arr[:, :, valid_slices]
    slc_grid = (resample_grid[0], resample_grid[1], resample_grid[2][valid_slices])
    proc_scan_arr, mask_out_4d, resize_grid = resizeScanAndMask(resamp_slc_arr,
                                                               input_mask_arr,
                                                               slc_grid,
                                                               out_size,
                                                               resize_method,
                                                               limitsM=limits)
    planC = pc.importScanArray(proc_scan_arr,
                                   resize_grid[0], resize_grid[1], resize_grid[2],
                                   modality, scan_num, planC)
    proc_scan_num = len(planC.scan) - 1

    scanList = [scan_num, resample_scan_num, proc_scan_num]

    return scanList, valid_slices, resize_grid, limits, planC


def normalize_data_HN_window(data):
    """Scale input intensities"""
    data[data < 1114 - 175] = 939   # 1114-175
    data[data > 1114 + 175] = 1289  # 1114+175
    data = (data - 939) / 175 - 1
    return (data)


def add_CT_offset(scan):
    """Offset input intensities (in HU) by fixed amount"""
    offset = 1024
    offset_scan = scan + offset
    return offset_scan


def postproc_and_import_seg(label_map, scan_list, planC, out_slices, resize_grid, limits):
    """ Post-process & import label map to planC """

    orig_scan_num = scan_list[0]
    resamp_scan_num = scan_list[1]
    orig_scan_size = planC.scan[orig_scan_num].getScanSize()


    # Convert label map to binary mask stack
    output_mask = label_to_bin(label_map)     #4-D

    #------------------------------------
    # Undo pre-processing transformations
    #------------------------------------

    # 1. Undo 2d cropping and resizing
    output_scan = None
    method = 'unpad2d'
    resamp_size = planC.scan[resamp_scan_num].getScanSize()
    resamp_out_size = [resamp_size[0], resamp_size[1], len(out_slices)]

    _, unpad_mask, unpad_grid = resizeScanAndMask(output_scan,
                                                  output_mask,
                                                  resize_grid,
                                                  resamp_out_size,
                                                  method,
                                                  limitsM=limits)
    # 2. Undo crop (slices)
    resamp_mask = np.full((resamp_size[0], resamp_size[1],
                          resamp_size[2], unpad_mask.shape[3]), 0)
    resamp_mask[:, :, out_slices, :] = unpad_mask

    #------------------------------------
    # Resample and post-process masks
    #------------------------------------
    num_components = 1
    replace_str_num = None
    label_map_out = np.zeros(orig_scan_size)
    proc_str_list = []

    # Loop over labels
    for label_idx in range(len(label_dict)):

        # Import mask to resampled scan
        str_name = output_str_names[label_idx]
        mask_idx = output_str_labels[label_idx] - 1
        output_mask = resamp_mask[:, :, :, mask_idx]
        planC = pc.importStructureMask(output_mask, resamp_scan_num,
                                       str_name, planC, replace_str_num)
        proc_str = len(planC.structure) - 1

        # Copy to original scan (undo resampling)
        planC = structure.copyToScan(proc_str, orig_scan_num, planC)
        del planC.structure[proc_str] #Delete intermediate (resampled) struct
        proc_str_list.append(proc_str) 

        # Post-process and replace input structure in planC
        proc_mask_3d, planC = structure.getLargestConnComps(proc_str, num_components,
                                                     planC, saveFlag=True,
                                                     replaceFlag=True,
                                                     procSructName=str_name)
        label_map_out[proc_mask_3d] = output_str_labels[label_idx]

    return planC, proc_str_list, label_map_out


def label_to_bin(label_map):
    """Convert label map to binary mask stack"""
    label_siz = np.shape(label_map)
    out_siz = label_siz + (num_labels,)
    bin_mask = np.zeros(out_siz)

    for label in range(num_labels):
        bin_mask[:, :, :, label] = label_map == (label + 1)

    return bin_mask


def write_file(mask, out_file, input_img):
    """ Write mask to NIfTI file """
    
    maskShift = np.flip(np.moveaxis(mask,[2, 1, 0], [0, 2, 1]),0)
    mask_img = sitk.GetImageFromArray(maskShift)

    # Copy information from input img
    mask_img.CopyInformation(input_img)

    # Write to file
    sitk.WriteImage(mask_img, out_file)

    return 0

def mask_to_DICOM(pt_id, model_name, output_dir, struct_nums, scan_num, planC):
    """ Export AI auto-segmentatiosn to DIOCM RTSTRUCTs """

    struct_file_name = f"{pt_id}_{model_name}_AI_seg.dcm"
    struct_file_path = os.path.join(output_dir, struct_file_name)
    series_description = "AI Generated"
    export_opts = {'seriesDescription': series_description}
    orig_indices = np.array([cerrScn.getScanNumFromUID(planC.structure[struct_num].assocScanUID, \
                                                   planC) for struct_num in struct_nums], dtype=int)
    structs_to_export = np.array(struct_nums)[orig_indices == scan_num]
    rtstruct_iod.create(structs_to_export, struct_file_path, planC, export_opts)

    return 0

def main(input_nii_path, session_path, output_path, DCMexportFlag=False):

    os.makedirs(output_path, exist_ok=True)

    # Create output and session dirs
    model_in_path = os.path.join(session_path, 'input_nii')
    model_out_path = os.path.join(session_path, 'output_nii')
    os.makedirs(session_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(model_in_path, exist_ok=True)
    os.makedirs(model_out_path, exist_ok=True)

    train_opt = TrainOptions().parse() 
    
    # Import NIfTI scan to planC
    file_name = os.path.basename(input_nii_path)    
    pt_id = file_name.split('.nii')[0]
    planC = pc.loadNiiScan(input_nii_path, imageType="CT SCAN")
    orig_img = sitk.ReadImage(input_nii_path)

    # Pre-process and export to NIfTI
    scan_list, valid_slices, proc_grid, limits, planC = process_image(planC)
    orig_scan_num = scan_list[0]
    proc_scan_num = scan_list[2]
    proc_scan_filename = f"{pt_id}_scan_3D.nii.gz"
    proc_scan_filepath = os.path.join(model_in_path, f"{pt_id}_scan_3D.nii.gz")
    planC.scan[proc_scan_num].saveNii(proc_scan_filepath)

    # Load pre-trained weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wrapper_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(wrapper_dir, "models")
    model_weights_path = os.path.join(model_dir, 'model_save')
    model = create_model(train_opt)
    model.load_CT_seg_A(model_weights_path)

    # Load processed image
    ("******* Filename *******")
    print(proc_scan_filename)
    ct_arr, sitk_img = load_file(proc_scan_filepath)
    ct_arr = np.flip(np.array(ct_arr),2)
    img_arr = add_CT_offset(ct_arr)
    print(ct_arr.shape)

    print("****** Shape *******")
    height, width, length = np.shape(img_arr)
    print("Input image size")
    print(np.shape(img_arr))

    print('****** Normalize ******')
    img_norm_arr = normalize_data_HN_window(img_arr)
    img_flip_norm_arr = img_norm_arr.transpose(2, 0, 1)
    img_flip_norm_arr = img_flip_norm_arr.reshape(img_flip_norm_arr.shape[0],1,
                                                      img_flip_norm_arr.shape[1],
                                                      img_flip_norm_arr.shape[2])

    print("****** Final scan shape ******")
    print(np.shape(img_flip_norm_arr))

    # originally added for comparison, make it zero, should work
    images = torch.tensor(img_flip_norm_arr, dtype=torch.float32)  
    labels = torch.zeros_like(images)                                               # [N,1,H,W]
    images_ct_val = torch.utils.data.TensorDataset(images, labels)
    print('Data set size: ', images.shape)
    print('****** Data loading complete ******')


    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device=="cuda":
      model.netSeg_A.to(device)
      model.netSeg_B.to(device)
    label_map = np.zeros((input_size, input_size, length),
                             dtype=np.uint8)

    print("****** Apply model ******")
    train_loader_c1 = torch.utils.data.DataLoader(images_ct_val,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0)

    with torch.no_grad():  # no grade calculation
        for i, (img, label) in enumerate(train_loader_c1):
            img = img.to(device)
            label = label.to(device)
            input_tensor = torch.cat([img, label], dim=1)
            model.set_test_input(input_tensor)
            tep_dice_loss, ori_img, seg, gt, image_numpy = model.net_G_A_A2B_Segtest_image()
            label_map[:, :, i] = seg
    
    
    print("****** Undo pre-processing transformations ******")
    planC, proc_str_num, proc_label_map = postproc_and_import_seg(label_map, scan_list, planC,
                                          valid_slices, proc_grid, limits)

    print("****** Export structure to NIfTI ******")
    label_out_file_name = f"{pt_id}_{model_arch}_AI_seg.nii.gz"
    label_out_file_path = os.path.join(output_path, label_out_file_name)
    write_file(proc_label_map, label_out_file_path, orig_img)

    if DCMexportFlag:
        # Export to DICOM
        mask_to_DICOM(pt_id, model_arch, output_path, proc_str_num, orig_scan_num, planC)


    return label_out_file_path, proc_str_num, planC

if __name__ == "__main__":
    DCMexportFlag = False
    if len(sys.argv) > 4:
        DCMexportFlag = sys.argv[4]
    main(sys.argv[1], sys.argv[2], sys.argv[3], DCMexportFlag)


