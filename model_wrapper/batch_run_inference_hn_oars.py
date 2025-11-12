import os
import batch_run_inference_hn_oars_dcm
import batch_run_inference_hn_oars_nii

def main(inputPath, sessionPath, outputPath):

    # Create output dir
    os.makedirs(outputPath, exist_ok=True)

    # Examine input directory contents
    contents = [os.path.join(path, f) for f in os.listdir(inputPath)]
    if not contents:
        print("Input directory is empty.")
        return

    # Identify input type
    dcmFlag = all(os.path.isdir(f) for f in contents)
    niiFlag = all(os.path.isfile(f) for f in contents)

    # Run batch auto-seg
    if dcmFlag:
        batch_run_inference_hn_oars_dcm.main(inputPath, sessionPath, outputPath)
    elif niiFlag:
        batch_run_inference_hn_oars_nii.main(inputPath, sessionPath, outputPath)

    return 0


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])