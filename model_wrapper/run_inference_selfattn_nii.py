import fnmatch
import glob
import os
import sys

import SimpleITK as sitk
import numpy as np
import torch.utils.data
from pathlib import Path

from cerr import plan_container as pc
from cerr.radiomics.preprocess import getResampledGrid, imgResample3D
from cerr.utils.ai_pipeline import getScanNumFromIdentifier
from cerr.utils.image_proc import resizeScanAndMask
from cerr.utils.mask import computeBoundingBox, getPatientOutline
from cerr.dataclasses import structure
from cerr.contour import rasterseg as rs

from models.models import create_model
from options.train_options import TrainOptions

opt = TrainOptions().parse()

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

    gridType = 'center'
    resampMethod = 'sitkBSpline'
    outputResV = [0.1, 0.1, 0]  # Output res: 1mm x 1mm in-plane

    resizeMethod = 'pad2d'
    outSizeV = [256, 256]
    inputMask3M = None

    outlineStructName = 'outline'
    intensityThreshold = -400  # Air intensity for outline detection

    # Get scan array
    scanNum = getScanNumFromIdentifier(identifier, planC)[0]
    xValsV, yValsV, zValsV = planC.scan[scanNum].getScanXYZVals()
    scan3M = planC.scan[scanNum].getScanArray()

    # 1. Resample
    [xResampleV, yResampleV, zResampleV] = getResampledGrid(outputResV,
                                                            xValsV, yValsV, zValsV,
                                                            gridType)
    resampScan3M = imgResample3D(scan3M,
                                     xValsV, yValsV, zValsV,
                                     xResampleV, yResampleV, zResampleV,
                                     resampMethod, inPlane=True)

    resampleGridS = [xResampleV, yResampleV, zResampleV]
    planC = pc.importScanArray(resampScan3M,
                                   resampleGridS[0], resampleGridS[1], resampleGridS[2],
                                   modality, scanNum, planC)
    resampleScanNum = len(planC.scan) - 1

    # 2. Extract patient outline
    replaceStrNum = None
    outline3M = getPatientOutline(resampScan3M, intensityThreshold)
    resampSizeV = outline3M.shape
    planC = pc.importStructureMask(outline3M, scanNum,
                                       outlineStructName,
                                       planC, replaceStrNum)

    # 3. Crop to patient outline on each slice
    sumSlices = np.sum(outline3M, axis=(0, 1))
    validSlicesV = np.where(sumSlices > 0)[0]
    numSlcs = len(validSlicesV)
    limitsM = np.zeros((numSlcs, 4))

    for slc in range(numSlcs):
        minr, maxr, minc, maxc, _, _, _ = computeBoundingBox( \
                outline3M[:, :, validSlicesV[slc]],
                is2DFlag=True)
        limitsM[slc, :] = [minr, maxr, minc, maxc]

    # 4. Resize to 256 x 256 in-plane
    resampSlc3M = resampScan3M[:, :, validSlicesV]
    slcGridS = (resampleGridS[0], resampleGridS[1], resampleGridS[2][validSlicesV])
    procScan3M, maskOut4M, resizeGridS = resizeScanAndMask(resampSlc3M,
                                                               inputMask3M,
                                                               slcGridS,
                                                               outSizeV,
                                                               resizeMethod,
                                                               limitsM=limitsM)
    planC = pc.importScanArray(procScan3M,
                                   resizeGridS[0], resizeGridS[1], resizeGridS[2], \
                                   modality, scanNum, planC)
    procScanNum = len(planC.scan) - 1

    scanList = [scanNum, resampleScanNum, procScanNum]

    return scanList, validSlicesV, resizeGridS, limitsM, planC


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


def postproc_and_import_seg(output_dir, scan_list, planC, out_slices, resize_grid, limits):
    """ Post-process & import label map to planC """

    orig_scan_num = scan_list[0]

    # Read AI-generated mask
    nii_glob = glob.glob(os.path.join(output_dir, '*.nii.gz'))
    mask_list = []
    for file_path in nii_glob:
        print('Importing ' + file_path + '...')
        masks_4d = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(output_dir, file_path)))
        masks_4d = np.moveaxis(masks_4d,[3, 2, 1, 0],[0, 1, 2, 3])
        if masks_4d.ndim == 3:
            temp = np.zeros(masks_4d.shape + (1,))
            temp[:, :, :, 0] = masks_4d
            masks_4d = temp
        mask_list.append(masks_4d)
    output_mask_4d = np.concatenate(mask_list, axis=-1)

    # Undo pre-processing transformations
    resamp_scan_num = scan_list[1]

    # 1. Undo 2d cropping and resizing
    output_scan = None
    method = 'unpad2d'
    resamp_size = planC.scan[resamp_scan_num].getScanSize()
    resamp_out_size = [resamp_size[0], resamp_size[1], len(out_slices)]

    _, unpad_mask_4d, unpad_grid = resizeScanAndMask(output_scan,
                                                     output_mask_4d,
                                                     resize_grid,
                                                     resamp_out_size,
                                                     method,
                                                     limitsM=limits)
    # 2. Undo outline crop (slices)
    resamp_mask_4d = np.full((resamp_size[0], resamp_size[1],
                              resamp_size[2], unpad_mask_4d.shape[3]), 0)
    resamp_mask_4d[:, :, out_slices, :] = unpad_mask_4d

    # Import to planC
    num_components = 1
    replace_str_num = None
    proc_str_list = []
    proc_mask_list = []
    # Loop over labels
    for label_idx in range(len(label_dict)):
        # Import mask to processed scan
        str_name = output_str_names[label_idx]
        mask_idx = output_str_labels[label_idx] - 1
        output_mask = resamp_mask_4d[:, :, :, mask_idx]
        planC = pc.importStructureMask(output_mask, resamp_scan_num,
                                       str_name, planC, replace_str_num)
        proc_str = len(planC.structure) - 1

        # Copy to original scan (undo resampling)
        planC = structure.copyToScan(proc_str, orig_scan_num, planC)
        proc_str_list.append(proc_str) #Since resamp str is eventually deleted

        # Post-process and replace input structure in planC
        proc_mask_3d, planC = structure.getLargestConnComps(proc_str, num_components,
                                                     planC, saveFlag=True,
                                                     replaceFlag=True,
                                                     procSructName=str_name)
        proc_mask_4d = np.full((np.shape(proc_mask_3d) + (1,)), 0)
        proc_mask_4d[:, :, :, 0] = proc_mask_3d
        proc_mask_list.append(proc_mask_4d)
        del planC.structure[proc_str]

    return planC, proc_str_list, proc_mask_list


def label_to_bin(label_map):
    """Convert label map to binary mask stack"""
    label_siz = np.shape(label_map)
    out_siz = label_siz + (num_labels,)
    bin_mask = np.zeros(out_siz)

    for label in range(num_labels):
        bin_mask[:, :, :, label] = label_map == (label + 1)

    return bin_mask


def write_file(mask, dir_name, file_name, input_img):
    """ Write mask to NIfTI file """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    out_file = os.path.join(dir_name, file_name)
    maskShift = np.moveaxis(mask,[3, 2, 1, 0], [0, 1, 2, 3])
    mask_img = sitk.GetImageFromArray(maskShift, isVector=False)

    # Copy information from input img
    origin_3d = list(input_img.GetOrigin())
    origin_4d = origin_3d + [0.0]  # Append 0.0 for the 4th dimension
    mask_img.SetOrigin(origin_4d)
    spacing_3d = list(input_img.GetSpacing())
    spacing_4d = spacing_3d + [1.0]
    mask_img.SetSpacing(spacing_4d)
    direction_3d = list(input_img.GetDirection())
    direction_4d = [direction_3d[0], direction_3d[1], direction_3d[2], 0.0,
                    direction_3d[3], direction_3d[4], direction_3d[5], 0.0,
                    direction_3d[6], direction_3d[7], direction_3d[8], 0.0,
                    0.0,             0.0,             0.0,             1.0]
    mask_img.SetDirection(direction_4d)

    sitk.WriteImage(mask_img, out_file)

def main(argv):

    input_nii_path = argv[1]
    session_path = argv[2]
    output_nii_path = argv[3]

    # Create output and session dirs
    model_in_path = os.path.join(session_path, 'input_nii')
    model_out_path = os.path.join(session_path, 'output_nii')
    os.makedirs(session_path, exist_ok=True)
    os.makedirs(output_nii_path, exist_ok=True)
    os.makedirs(model_in_path, exist_ok=True)
    os.makedirs(model_out_path, exist_ok=True)

    # Import NIfTI scan to planC
    file_name = os.listdir(input_nii_path)[0]
    file_path = os.path.join(input_nii_path,file_name)
    pt_id = Path(Path(input_nii_path).stem).stem
    planC = pc.loadNiiScan(file_path, imageType="CT SCAN")
    orig_img = sitk.ReadImage(file_path)

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
    model = create_model(opt)
    model.load_CT_seg_A(model_weights_path)

    # Load processed image
    ("*******Filename*******")
    print(proc_scan_filename)
    ct_arr, sitk_img = load_file(proc_scan_filepath)
    ct_arr = np.array(ct_arr)
    img_arr = add_CT_offset(ct_arr)
    print(ct_arr.shape)

    print("******Shape*******")
    height, width, length = np.shape(img_arr)
    print("Input image size")
    print(np.shape(img_arr))

    print('******Normalize******')
    img_norm_arr = normalize_data_HN_window(img_arr)
    img_flip_norm_arr = img_norm_arr.transpose(2, 0, 1)
    img_flip_norm_arr = img_flip_norm_arr.reshape(img_flip_norm_arr.shape[0],1,
                                                      img_flip_norm_arr.shape[1],
                                                      img_flip_norm_arr.shape[2])

    print("******Final scan shape******")
    print(np.shape(img_flip_norm_arr))
    # originally added for comparison, make it zero, should work
    label_arr = np.zeros(shape=np.shape(img_flip_norm_arr))
    #print(np.shape(label_arr))

    images_ct_val = np.concatenate((np.array(img_flip_norm_arr),
                                        np.array(label_arr)), 1)
    label_map = np.zeros((input_size, input_size, length),
                             dtype=np.uint8)

    print('Data set size: ', img_flip_norm_arr.shape)
    print('******Data loading complete******')

    print("****** Apply model ******")
    train_loader_c1 = torch.utils.data.DataLoader(images_ct_val,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1)

    with torch.no_grad():  # no grade calculation
        for i, (data_val) in enumerate(zip(train_loader_c1)):
            model.set_test_input(data_val)
            tep_dice_loss, ori_img, seg, gt, image_numpy = model.net_G_A_A2B_Segtest_image()
            label_map[:, :, i] = seg

    # Convert 3D label map to 4D stack of binary masks
    bin_mask = label_to_bin(label_map)

    print("****** Export mask to NIfTI ******")
    mask_filename = f"{pt_id}_model_out.nii.gz"
    write_file(bin_mask, model_out_path, mask_filename, sitk_img)

    print("****** Undo pre-processing transformations ******")
    planC, proc_str_num, proc_mask_list = postproc_and_import_seg(model_out_path, scan_list, planC,
                                          valid_slices, proc_grid, limits)

    print("****** Export structure to NIfTI ******")
    output_files = []
    for mask_num in range(len(proc_mask_list)):
        struct_file_name = f"{pt_id}_{output_str_names[mask_num]}_AI_seg.nii.gz"
        write_file(proc_mask_list[mask_num], output_nii_path, struct_file_name, orig_img)

    return output_files, proc_str_num, planC

if __name__ == "__main__":
    main(sys.argv)


