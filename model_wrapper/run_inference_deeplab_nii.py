import os
import shutil
import sys

import SimpleITK as sitk
import numpy as np
from skimage.morphology import remove_small_objects

from cerr import plan_container as pc
from cerr.contour import rasterseg as rs
from cerr.dataclasses import structure as cerrStr
from cerr.dcm_export import rtstruct_iod
from cerr.utils import mask
from cerr.utils.ai_pipeline import getScanNumFromIdentifier
from cerr.utils.image_proc import transformScan
from cerr.utils.statistics import prctile, round

import run_fuse_inference_chewing_nii
import run_fuse_inference_constrictor_nii
import run_fuse_inference_larynx_nii


def postProcessChew(mask3M):
    "Post-processing of AI segmentations of chewing structures"

    maskSiz = mask3M.shape[0]
    scale = 512 / maskSiz

    # Morphological closing element
    radius = int(np.floor(5 / scale))

    # Smoothing filter dimensions
    filtSize = np.floor(3 / scale)
    filtSize = int(max(1, 2 * np.floor(filtSize // 2) - 1))  # Nearest odd value

    # Min size for connected components
    minSizeIsland = np.floor(10 / scale ** 2)
    minSizeCC = np.floor(1000 / scale ** 2)

    procMask3M = mask3M.copy()
    slicesV = np.where(np.sum(np.sum(mask3M.astype(float), axis=0), axis=0) > 0)[0]
    maskSlc3M = mask3M[:, :, slicesV]

    strElement = mask.createStructuringElement(radius, [1, 1, 1], shape='sphere')
    fillMask3m = mask.morphologicalClosing(maskSlc3M, strElement)

    # Remove islands  & retain largest CC
    for s in range(fillMask3m.shape[2]):
        slcMask = fillMask3m[:, :, s]
        temp = remove_small_objects(slcMask.astype(bool),
                                                   int(minSizeIsland),
                                                   connectivity=8)
        
        strMaskM = mask.largestConnComps(temp, 1,
                                         minSize=minSizeIsland, dim=2)
        fillMask3m[:, :, s] = strMaskM

    strMask3M = mask.largestConnComps(fillMask3m, 1,
                                      minSize=minSizeCC, dim=3)

    # Smooth
    if strMask3M.shape[2] > 1:
        filtRadius = int((filtSize-1)/2)
        smoothedLabel3M = mask.blurring(strMask3M, filtRadius, filtType='box')
        strMask3M = smoothedLabel3M > 0.4

    procMask3M[:, :, slicesV] = strMask3M

    return procMask3M


def postProcessLar(mask3M):
    "Post-processing of AI segmentations of the larynx"

    maskSiz = mask3M.shape[0]
    scale = 512 / maskSiz

    # Morphological closing element
    radius = int(np.floor(3 / scale))

    # Smoothing filter dimensions
    filtSize = np.floor(5 / scale)
    filtSize = int(max(1, 2 * np.floor(filtSize // 2) - 1))  # Nearest odd value

    # Min size for connected components
    minSizeIsland = np.floor(10 / scale ** 2)
    minSizeCC = np.floor(1000 / scale ** 2)

    procMask3M = mask3M
    slicesV = np.where(np.sum(np.sum(mask3M.astype(float), axis=0), axis=0) > 0)[0]
    maskSlc3M = mask3M[:, :, slicesV]

    # Fill holes
    strElement = mask.createStructuringElement(radius, [1, 1, 1], shape='sphere')
    fillMask3m = mask.morphologicalClosing(maskSlc3M, strElement)

    # Remove islands  & retain largest CC
    for s in range(fillMask3m.shape[2]):
        slcMask = fillMask3m[:, :, s]
        fillMask3m[:, :, s] = remove_small_objects(slcMask.astype(bool),
                                                   int(minSizeIsland),
                                                   connectivity=8)
        temp = fillMask3m[:, :, s]
        strMaskM = mask.largestConnComps(temp, 1,
                                         minSize=minSizeIsland, dim=2)
        fillMask3m[:, :, s] = strMaskM

    strMask3M = mask.largestConnComps(fillMask3m, 1,
                                      minSize=minSizeCC, dim=3)

    # Smooth
    if strMask3M.shape[2] > 1:
        filtRadius = int((filtSize-1)/2)
        smoothedLabel3M = mask.blurring(strMask3M, filtRadius, filtType='box')
        strMask3M = smoothedLabel3M > 0.5

    procMask3M[:, :, slicesV] = strMask3M

    return procMask3M


def postProcessCM(mask3M):
    "Post-processing of AI segmentations of the constrictor"

    maskSiz = mask3M.shape[0]
    scale = 512 / maskSiz

    # Morphological closing element
    radius1 = int(np.floor(4 / scale))
    radius2 = int(np.floor(2 / scale))

    # Smoothing filter dimensions
    filtSize = np.floor(5 / scale)
    filtSize = int(max(1, 2 * np.floor(filtSize // 2) - 1))  # Nearest odd value

    # Min size for connected components
    minSizeIsland = np.floor(20 / scale ** 2)
    minSizeCC = np.floor(50 / scale ** 2)

    procMask3M = mask3M
    slicesV = np.where(np.sum(np.sum(mask3M.astype(float), axis=0), axis=0) > 0)[0]
    maskSlc3M = mask3M[:, :, slicesV]

    # Fill holes and retain largest CC
    strElement = mask.createStructuringElement(radius1, [1, 1, 1], shape='sphere')
    fillMask3m = mask.morphologicalClosing(maskSlc3M, strElement)
    fillMask3m = mask.largestConnComps(fillMask3m, 1,
                                       minSize=minSizeCC, dim=3)

    # Fill holes and remove islands
    strElement2 = mask.createStructuringElement(radius2, [1, 1, 1],
                                           shape='disk')
    for s in range(fillMask3m.shape[2]):
        slcMask = fillMask3m[:, :, s]
        labelM = mask.morphologicalClosing(slcMask, strElement2)

        strMaskM = mask.largestConnComps(labelM, 1,
                                         minSize=minSizeIsland, dim=2)
        fillMask3m[:, :, s] = strMaskM

    # Smooth
    if fillMask3m.shape[2] > 1:
        filtRadius = int((filtSize-1)/2)
        smoothedLabel3M = mask.blurring(fillMask3m, filtRadius, filtType='box')
        fillMask3m = smoothedLabel3M > 0.5

    procMask3M[:, :, slicesV] = fillMask3m

    return procMask3M


def postProc(selectedMethod, *params):
    """
    Apply post-processing for specified model
    """
    methodDict = {"chew": postProcessChew,
                  "larynx": postProcessLar,
                  "cm": postProcessCM}
    fnHandle = methodDict[selectedMethod]
    procMask3M = fnHandle(*params)
    return procMask3M


def preProcChew(planC, niiDir):
    """
    Localize chewing structures on HN CT image and save cropped NIfTI image to niiDir
    """
    identifier = {"imageType": "CT SCAN"}
    scanIndices = getScanNumFromIdentifier(identifier, planC, False)

    # Pre-processing parameters
    outlineThreshold = -400
    headSizeLimitCm = 23
    views = ['axial', 'sagittal', 'coronal']
    regionName = 'chew_crop'

    # -----------------------------------------------
    #          Localize chewing structures
    # -----------------------------------------------
    scanIndex = scanIndices[0]
    scanArray = planC.scan[scanIndex].getScanArray()
    scanGrid = planC.scan[scanIndex].getScanXYZVals()
    zValues = scanGrid[2]

    # Get mask of pt outline
    outerMask = mask.getPatientOutline(scanArray, outlineThreshold)
    # Get S-I limits
    mins = next((i for i, val in enumerate(outerMask.sum(axis=(0, 1))) \
                 if val > 0), 1)
    maxs = outerMask.shape[2] - 1
    # Get head extent
    zStart = zValues[mins]
    zVals = zValues[mins:]
    zDiffs = np.array([z - zStart for z in zVals])
    endSlc = np.argmin(abs(zDiffs - headSizeLimitCm))
    endSlc = endSlc + mins
    if maxs < endSlc:
        SIextent = zVals[maxs] - zVals[mins]
        print(f"Warning: Input scan S-I extent = {SIextent} cm. \
                Model performance on cropped scans (S-I extent <\
                {headSizeLimitCm} cm) is untested.")
        endSlc = min(endSlc, maxs)
    # Get A, P extents
    minrSlc = [float('nan')] * outerMask.shape[2]
    maxrSlc = [float('nan')] * outerMask.shape[2]
    mincSlc = [float('nan')] * outerMask.shape[2]
    maxcSlc = [float('nan')] * outerMask.shape[2]
    for n in range(endSlc):
        maskSlc = outerMask[:, :, n]
        if maskSlc.any():
            minrSlc[n], maxrSlc[n], mincSlc[n], maxcSlc[n], __, __, __ = \
                mask.computeBoundingBox(maskSlc, is2DFlag=True)
    minr = round(prctile(np.array(minrSlc), 5)).astype(int)
    maxr = round(np.nanmedian(np.array(maxrSlc)))
    width = maxr - minr + 1
    maxr = round(maxr - 0.25 * width).astype(int)
    # Get L, R extents
    slcrMin = np.nanargmin(abs(minrSlc - minr))
    minc = int(mincSlc[slcrMin])
    maxc = int(maxcSlc[slcrMin])
    # Return bounding box
    bbox = np.zeros_like(outerMask, dtype=bool)
    bbox[minr:maxr, minc:maxc, mins:maxs] = True
    planC = pc.importStructureMask(bbox, scanIndex, regionName, planC)
    croppedScan = scanArray[minr:maxr + 1, minc:maxc + 1, mins:maxs + 1]
    croppedGridS = (scanGrid[1][minr:maxr + 1],
                    scanGrid[0][minc:maxc + 1],
                    scanGrid[2][mins:maxs + 1])

    # -----------------------------------------
    # Transform orientation
    # ------------------------------------------
    for orientation in views:
        cropScanView, __, cropGridViewS = transformScan(croppedScan, None,
                                                        croppedGridS, orientation)
        # Export to NIfTI
        modality = 'CT'
        scanDir = os.path.join(niiDir, orientation)
        os.makedirs(scanDir, exist_ok=True)
        scanFilename = os.path.join(scanDir, 'scan.nii.gz')
        planC = pc.importScanArray(cropScanView, cropGridViewS[0],
                                   cropGridViewS[1], cropGridViewS[2],
                                   modality, scanIndex, planC)
        procScanNum = len(planC.scan) - 1
        planC.scan[procScanNum].saveNii(scanFilename)

    scanBounds = [minr, maxr, minc, maxc, mins, maxs]

    return scanIndex, scanBounds, planC


def preProcLar(planC, scanIndex, niiDir):
    """
    Localize larynx on HN CT image and save cropped NIfTI image to niiDir
    """

    chewingStrList = ["Left masseter", "Right masseter",
                      "Left medial pterygoid", "Right medial pterygoid"]
    modality = 'CT'

    strList = [str.structureName for str in planC.structure]

    scanArray = planC.scan[scanIndex].getScanArray()
    origSize = scanArray.shape
    scanGrid = planC.scan[scanIndex].getScanXYZVals()

    # ----------------------------
    # Crop scan around larynx
    # -----------------------------
    chewRegionName = 'chew_crop'
    regionName = 'larynx_crop'
    unionMask3M = np.full(origSize, False)
    for strName in chewingStrList:
        strNum = cerrStr.getMatchingIndex(strName, strList, matchCriteria='exact')[0]
        strMask3M = rs.getStrMask(strNum, planC)
        unionMask3M = unionMask3M | strMask3M
    minr, __, minc, maxc, mins, __, __ = mask.computeBoundingBox(unionMask3M)
    strNum = cerrStr.getMatchingIndex(chewRegionName,
                                      strList, matchCriteria='exact')[0]
    outMask3M = rs.getStrMask(strNum, planC)
    __, maxr, __, __, __, maxs, __ = mask.computeBoundingBox(outMask3M)
    # Return bounding box
    bbox = np.zeros_like(outMask3M, dtype=bool)
    bbox[minr:maxr, minc:maxc, mins:maxs] = True
    planC = pc.importStructureMask(bbox, scanIndex, regionName, planC)
    croppedScan = scanArray[minr:maxr + 1, minc:maxc + 1, mins:maxs + 1]
    croppedGridS = (scanGrid[1][minr:maxr + 1],
                    scanGrid[0][minc:maxc + 1],
                    scanGrid[2][mins:maxs + 1])

    # -------------------------
    # Transform orientation
    # ---------------------------
    views = ['axial', 'sagittal', 'coronal']
    for orientation in views:
        cropScanView, __, cropGridViewS = transformScan(croppedScan,
                                                        None, croppedGridS,
                                                        orientation)

        # #Export to NIfTI
        scanDir = os.path.join(niiDir, orientation)
        os.makedirs(scanDir, exist_ok=True)
        scanFilename = os.path.join(scanDir, 'scan.nii.gz')
        planC = pc.importScanArray(cropScanView, cropGridViewS[0],
                                   cropGridViewS[1], cropGridViewS[2],
                                   modality, scanIndex, planC)
        procScanNum = len(planC.scan) - 1
        planC.scan[procScanNum].saveNii(scanFilename)

    scanBounds = [minr, maxr, minc, maxc, mins, maxs]

    return scanIndex, scanBounds, planC


def preProcCM(planC, scanIndex, niiDir):
    """
    Localize constrictor on HN CT image and save cropped NIfTI image to niiDir
    """

    # Crop scan around constrictor
    tol_r = 30
    tol_s = 15
    larynxRegionName = 'larynx_crop'
    modality = 'CT'

    scanArray = planC.scan[scanIndex].getScanArray()
    scanGrid = planC.scan[scanIndex].getScanXYZVals()
    strList = [str.structureName for str in planC.structure]

    strNum = cerrStr.getMatchingIndex(larynxRegionName, strList,
                                      matchCriteria='exact')[0]
    outMask3M = rs.getStrMask(strNum, planC)
    minr, __, minc, maxc, mins, __, __ = mask.computeBoundingBox(outMask3M)
    strNum = cerrStr.getMatchingIndex('Larynx', strList,
                                      matchCriteria='exact')[0]
    larMask3M = rs.getStrMask(strNum, planC)
    __, maxr, __, __, __, maxs, __ = mask.computeBoundingBox(larMask3M)
    maxr = min(maxr + tol_r, larMask3M.shape[0])
    maxs = min(maxs + tol_s, larMask3M.shape[2])

    croppedScan = scanArray[minr:maxr + 1, minc:maxc + 1, mins:maxs + 1]
    croppedGridS = (scanGrid[1][minr:maxr + 1],
                    scanGrid[0][minc:maxc + 1],
                    scanGrid[2][mins:maxs + 1])

    # Transform orientation
    views = ['axial', 'sagittal', 'coronal']
    for orientation in views:
        cropScanView, __, cropGridViewS = transformScan(croppedScan, None,
                                                        croppedGridS, orientation)

        # #Export to NIfTI
        scanDir = os.path.join(niiDir, orientation)
        os.makedirs(scanDir, exist_ok=True)
        scanFilename = os.path.join(scanDir, 'scan.nii.gz')
        planC = pc.importScanArray(cropScanView, cropGridViewS[0],
                                   cropGridViewS[1], cropGridViewS[2],
                                   modality, scanIndex, planC)
        procScanNum = len(planC.scan) - 1
        planC.scan[procScanNum].saveNii(scanFilename)

    scanBounds = [minr, maxr, minc, maxc, mins, maxs]

    return scanIndex, scanBounds, planC


def exportSegs(planC, scanIndex, scanBounds, modelName, segPath, outputPath, ptID):
    """Import NIfTI segmentation masks, apply post-processing, and export to DICOM """
    if modelName == 'chew':
        strToLabelMap = {1: "Left masseter", 2: "Right masseter",
                         3: "Left medial pterygoid",
                         4: "Right medial pterygoid"}
    elif modelName == 'larynx':
        strToLabelMap = {1: "Larynx"}
    elif modelName == 'cm':
        strToLabelMap = {1: "Pharyngeal constrictor"}

    scanArray = planC.scan[scanIndex].getScanArray()
    origSize = scanArray.shape

    outputStrNames = list(strToLabelMap.values())
    maskFile = os.path.join(segPath, 'mask.nii.gz')
    outputMaskImg = sitk.ReadImage(maskFile)
    outputMask4M = np.moveaxis(sitk.GetArrayFromImage(outputMaskImg),
                               0, 2)
    if outputMask4M.ndim == 3:
        temp = np.zeros(outputMask4M.shape+(1,))
        temp[:,:,:,0] = outputMask4M
        outputMask4M = temp

    # Reverse transformations
    fullMask3M = np.full(origSize, fill_value=0)
    replaceStrNum = None
    procStrV = []

    for labelIdx in range(len(strToLabelMap)):
        # Import mask to planC
        strName = outputStrNames[labelIdx]
        outputMask = outputMask4M[:, :, :, labelIdx]
        fullMask3M[scanBounds[0]:scanBounds[1] + 1, \
        scanBounds[2]:scanBounds[3] + 1, \
        scanBounds[4]:scanBounds[5] + 1] = outputMask

        # Post-process
        procmask3M = postProc(modelName, fullMask3M)
        planC = pc.importStructureMask(procmask3M, scanIndex,
                                       strName, planC, replaceStrNum)
        procStr = len(planC.structure) - 1
        procStrV.append(procStr)

    # Export segmentations to DICOM
    structFileName = ptID + '_' + modelName + '_AI_seg.dcm'
    structFilePath = os.path.join(outputPath, structFileName)
    seriesDescription = "AI Generated"
    exportOpts = {'seriesDescription': seriesDescription}
    rtstruct_iod.create(procStrV, structFilePath, planC, exportOpts)

    return planC


def main(inputPath, sessionpath, outputPath):

    # Create output and session dirs
    os.makedirs(sessionpath, exist_ok=True)
    os.makedirs(outputPath, exist_ok=True)

    # Read nii image
    ptID = os.path.basename(inputPath)
    planC = pc.loadNiiScan(inputPath, imageType="CT SCAN")

    # Segment chewing structures
    modelName = 'chew'
    scanIndex, scanBounds, planC = preProcChew(planC, sessionpath)
    tempNiiPath = os.path.join(sessionpath, 'chew_out_nii')
    run_fuse_inference_chewing_nii.main(sessionpath, tempNiiPath)
    planC = exportSegs(planC, scanIndex, scanBounds,
                       modelName, tempNiiPath, outputPath, ptID)

    # Segment larynx
    modelName = 'larynx'
    scanIndex, scanBounds, planC = preProcLar(planC, scanIndex, sessionpath)
    tempNiiPath = os.path.join(sessionpath, 'larynx_out_nii')
    run_fuse_inference_larynx_nii.main(sessionpath, tempNiiPath)
    planC = exportSegs(planC, scanIndex, scanBounds, modelName,
                       tempNiiPath, outputPath, ptID)

    # Segment pharyngeal constrictor
    modelName = 'cm'
    scanIndex, scanBounds, planC = preProcCM(planC, scanIndex, sessionpath)
    tempNiiPath = os.path.join(sessionpath, 'cm_out_nii')
    run_fuse_inference_constrictor_nii.main(sessionpath, tempNiiPath)
    planC = exportSegs(planC, scanIndex, scanBounds, modelName,
                       tempNiiPath, outputPath, ptID)

    # Clear session dir
    shutil.rmtree(sessionpath)
    return planC


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])