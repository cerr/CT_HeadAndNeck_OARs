import sys
import os
from model_wrapper import run_inference_deeplab, run_inference_selfattn

def main(inputPath, sessionPath, outputPath):

    # Create output dir
    os.makedirs(outputPath, exist_ok=True)

    # Identify input type
    if os.path.isfile(inputPath) and \
            (inputPath.endswith('.nii') or inputPath.endswith('.nii.gz')):
        dcmFlag = False #NIfTI
    elif os.path.isdir(inputPath):
        dcmFlag = True
    else:
        raise ValueError('Invalid input path ', inputPath)

    # Run auto-seg
    run_inference_deeplab.main(inputPath, sessionPath, outputPath, DCMexportFlag=dcmFlag)
    run_inference_selfattn.main(inputPath, sessionPath, outputPath, DCMexportFlag=dcmFlag)
    
    return 0


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
