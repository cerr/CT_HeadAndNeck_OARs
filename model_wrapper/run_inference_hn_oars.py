import sys
import os
from model_wrapper import run_inference_deeplab_dcm, run_inference_selfattn_dcm, run_inference_selfattn_nii, \
    run_inference_deeplab_nii


def main(inputPath, sessionPath, outputPath):

    # Create output dir
    os.makedirs(outputPath, exist_ok=True)

    # Identify input type
    if os.path.isfile(inputPath) and \
            (inputPath.endswith('.nii') or inputPath.endswith('.nii.gz')):
        niiFlag = True
    elif os.path.isdir(inputPath):
        dcmFlag = True
    else:
        raise ValueError('Invalid input path ', inputPath)

    # Run batch auto-seg
    if dcmFlag:
        run_inference_hn_oars_dcm.main(inputPath, sessionPath, outputPath)

    elif niiFlag:
        run_inference_deeplab_nii.main(inputPath, sessionPath, outputPath)
        run_inference_selfattn_nii.main(inputPath, sessionPath, outputPath)

    return 0


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])