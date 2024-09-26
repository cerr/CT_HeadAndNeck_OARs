import fnmatch
import os
import sys

import numpy as np
import SimpleITK as sitk
import torch.utils.data
from models.models import create_model
from options.train_options import TrainOptions

opt = TrainOptions().parse()

# Input image dimensions
input_size = 256

# No. output structures
num_labels = 11 

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

def label_to_bin(label_map):
    """Convert label map to binary mask stack"""
    label_siz = np.shape(label_map)
    out_siz = label_siz + (num_labels,)
    bin_mask = np.zeros(out_siz)

    for label in range(num_labels):
        bin_mask[:,:,:,label] = label_map==(label+ 1)

    return bin_mask

def write_file(mask, dir_name, file_name, input_img):
    """ Write mask to NIfTI file """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    out_file = os.path.join(dir_name, file_name + '.nii.gz')
    mask = np.flip(mask,0)
    mask_img = sitk.GetImageFromArray(mask, isVector=False)

    # Copy information from input img
    origin3d = list(input_img.GetOrigin())
    origin4d = origin3d + [0.0]  # Append 0.0 for the 4th dimension
    mask_img.SetOrigin(origin4d)
    spacing3d = list(input_img.GetSpacing())
    spacing4d = spacing3d + [1.0]
    mask_img.SetSpacing(spacing4d)
    direction3d = list(input_img.GetDirection())
    direction4d = [ direction3d[0], direction3d[1], direction3d[2], 0.0,
                    direction3d[3], direction3d[4], direction3d[5], 0.0,
                    direction3d[6], direction3d[7], direction3d[8], 0.0,
                    0.0,            0.0,            0.0,            1.0]
    mask_img.SetDirection(direction4d)

    sitk.WriteImage(mask_img, out_file)

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

def main(argv):
    num_args = len(argv)
    if num_args == 1:  # container
        input_nii_path = '/scratch/inputNii/'
        output_nii_path = '/scratch/outputNii/'
    else:
        input_nii_path = sys.argv[1]
        output_nii_path = sys.argv[2]

    # Load pre-trained weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wrapper_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(wrapper_dir, "models")
    model_weights_path = os.path.join(model_dir, 'model_save')
    model = create_model(opt)
    model.load_CT_seg_A(model_weights_path)

    # Identify input files
    keyword = 'SCAN'
    files = find('*.nii', input_nii_path)
    if len(files) == 0:
        files = find('*.nii.gz', input_nii_path)
        if len(files) == 0:
            raise Exception("Invalid input file format.")

    # Loop over files
    for filename in files:
        print("*******Filename*******")
        print(filename)
        __, infilename = os.path.split(filename)
        ct_arr, sitk_img = load_file(filename)
        ct_arr = np.array(ct_arr)

        print('******Load scan array******')
        print(ct_arr.shape)
        img_arr = add_CT_offset(ct_arr)

        print("******shape*******")
        height, width, length = np.shape(img_arr)
        print("input scan shape")
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
        print("dummy label shape")
        print(np.shape(label_arr))

        images_ct_val = np.concatenate((np.array(img_flip_norm_arr),
                                        np.array(label_arr)), 1)
        label_map = np.zeros((length, input_size, input_size),
                             dtype=np.uint8)

        print('data set size is: ', img_flip_norm_arr.shape)
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
                label_map[i, :, :] = seg

    # Convert 3D label map to 4D stack of binary masks
    bin_mask = label_to_bin(label_map)

    print("****** Export mask to NIfTI ******")
    mask_filename = infilename.replace(keyword, 'file')
    write_file(bin_mask, output_nii_path, mask_filename, sitk_img)
    print("DONE")


if __name__ == "__main__":
    main(sys.argv)
