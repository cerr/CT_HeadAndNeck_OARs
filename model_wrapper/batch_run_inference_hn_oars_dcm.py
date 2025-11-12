import os
import shutil
import sys

import run_inference_deeplab_dcm
import run_inference_selfattn_dcm


def main(inputPath, sessionPath, outputPath):

    # Create output dir
    os.makedirs(outputPath, exist_ok=True)

    # Loop over pts
    ptList = os.listdir(inputPath)
    count = 0 
    for pt in ptList:
        count = count + 1
        print('Segmenting dataset {} of {}'.format(count, len(ptList)))
        ptDir = os.path.join(inputPath, pt)
        ptSessionDir = os.path.join(sessionPath, pt)
        run_inference_deeplab_dcm.main(ptDir, ptSessionDir, outputPath)
        run_inference_selfattn_nii.main(ptDir, ptSessionDir, outputPath)

    return 0


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])

