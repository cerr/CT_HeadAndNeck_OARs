import os
import shutil
import sys

import run_inference_deeplab_nii
import run_inference_selfattn_nii


def main(inputPath, sessionPath, outputPath, DCMexportFlag=False):

    # Create output dir
    os.makedirs(outputPath, exist_ok=True)

    # Loop over pts
    fileList = os.listdir(inputPath)
    count = 0 
    for fileName in fileList:
        count = count + 1
        print('Segmenting file {} of {}'.format(count, len(fileList)))
        ptID = fileName.split('.')[0]
        filePath = os.path.join(inputPath, fileName)
        ptSessionDir = os.path.join(sessionPath, ptID)
        run_inference_selfattn_nii.main(filePath, ptSessionDir, outputPath)

    return 0


if __name__ == '__main__':
    DCMexportFlag = False
    if len(sys.argv) > 4:
        DCMexportFlag = sys.argv[4]
    main(sys.argv[1], sys.argv[2], sys.argv[3], DCMexportFlag)

