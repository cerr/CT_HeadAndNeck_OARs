import sys
import os
from pathlib import Path

import run_inference_deeplab
import run_inference_selfattn

def main(inputPath, sessionPath, outputPath):

    # Create output dir
    os.makedirs(outputPath, exist_ok=True)

    # Examine input directory contents
    contents = [os.path.join(inputPath, f) for f in os.listdir(inputPath)]
    if not contents:
        print("Input directory is empty.")
        return

    # Identify input type
    dcmFlag = all(os.path.isdir(f) for f in contents)
    niiFlag = all(os.path.isfile(f) for f in contents)

    # Run batch auto-seg
    if dcmFlag:
        # Loop over pts
        count = 0
        for ptDir in contents:
            count = count + 1
            print('Segmenting dataset {} of {}'.format(count, len(contents)))
            ptID = Path(Path(ptDir).stem).stem
            ptSessionDir = os.path.join(sessionPath, ptID)
            run_inference_deeplab.main(ptDir, ptSessionDir, outputPath)
            run_inference_selfattn.main(ptDir, ptSessionDir, outputPath)

    elif niiFlag:
        fileList = os.listdir(inputPath)
        count = 0
        for fileName in fileList:
            count = count + 1
            print('Segmenting file {} of {}'.format(count, len(fileList)))
            ptID = os.path.basename(inputPath)
            filePath = os.path.join(inputPath, fileName)
            ptSessionDir = os.path.join(sessionPath, ptID)
            run_inference_deeplab.main(filePath, ptSessionDir, outputPath)
            run_inference_selfattn.main(filePath, ptSessionDir, outputPath)

    return 0


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])