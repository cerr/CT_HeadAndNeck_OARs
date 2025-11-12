# Author: Aditi Iyer
# Email: iyera@mskcc.org
# Date: Oct 25, 2019
# Description: Script to generate segmentations of the larynx by fusing
#              probability maps from three different views.
# Architecture :  Deeplab v3+ with resnet backbone
# Usage: python run_fuse_inference_larynx_nii.py [inputpath] [outputpath]
# Output: 2D masks saved as NIfTI files to output folder

import gzip, sys
from time import process_time

from dataloaders.custom_dataset import *
from modeling.deeplab import *
from skimage.transform import resize
from torch.utils.data import DataLoader
from tqdm import tqdm

def labelToBin(labelMap, numStructs):
    """Convert label map to binary mask stack"""
    labelSiz = np.shape(labelMap)
    outSiz = labelSiz + (numStructs,)
    binMask = np.zeros(outSiz)
    for label in range(numStructs):
        binMask[:, :, :, label] = labelMap == (label + 1)
    return binMask


def writeFile(mask, dirName, inputImg):
    """ Write mask to NIfTI file """
    fileName = 'mask.nii.gz'
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    outFile = os.path.join(dirName, fileName)
    mask = np.transpose(mask, (2, 0, 1, 3))
    maskImg = sitk.GetImageFromArray(mask)
    maskImg.CopyInformation(inputImg)
    sitk.WriteImage(maskImg, outFile)


class Larynx(object):
    def __init__(self, inputDir, view):

        self.inputDir = inputDir
        self.view = view
        self.nClass = 2
        self.cropSize = 321
        self.batchSize = 1

        cDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        if view == 'axial':
            self.mean = (0.3416, 0.3416, 0.3416)
            self.std = (0.1889, 0.1889, 0.1889)
            self.dataPath =  os.path.join(self.inputDir,'axial')
            self.modelPath = os.path.join(cDir, 'models', 'Larynx_Ax_model_state.pth.gz')
        elif view == 'sagittal':
            self.mean = (0.3157, 0.3157, 0.3157)
            self.std = (0.1803, 0.1803, 0.1803)
            self.modelPath = os.path.join(cDir, 'models', 'Larynx_Sag_model_state.pth.gz')
            self.dataPath = os.path.join(self.inputDir,'sagittal')
        elif view == 'coronal':
            self.mean = (0.3233, 0.3233, 0.3233)
            self.std = (0.1917, 0.1917, 0.1917)
            self.modelPath = os.path.join(cDir, 'models', 'Larynx_Cor_model_state.pth.gz')
            self.dataPath = os.path.join(self.inputDir,'coronal')
        else:
            raise ValueError('Invalid input view = %s' %(view))


        # Define dataloadr
        kwargs = {'num_workers': 0, 'pin_memory': True}
        testSet = struct(self)
        self.testLoader = DataLoader(testSet, batch_size=self.batchSize,
                                     shuffle=False, drop_last=False, **kwargs)

        # Define network
        print('Loading network...')
        t0 = process_time()
        self.model = DeepLab(num_classes=self.nClass,
                             backbone='resnet',
                             output_stride=16,
                             sync_bn=False,
                             freeze_bn=False)
        print("%.1f" %(process_time() - t0) + ' s')

        # Using CUDA
        print('Loading model weights...')
        t1 = process_time()
        if torch.cuda.device_count() and torch.cuda.is_available():
            print('Using GPU')
            self.cuda = True
            print('GPU device count: ', torch.cuda.device_count())
            device = torch.device("cuda:0")
            with gzip.open(self.modelPath, "rb") as f:
                state_dict = torch.load(f, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(device)
        else:
            self.cuda = False
            print('Using CPU')
            device = torch.device('cpu')
            with gzip.open(self.modelPath, "rb") as f:
                state_dict = torch.load(f, map_location=device)
            self.model.load_state_dict(state_dict)

        print("%.1f" %(process_time() - t1) + ' s')
        print('Loaded.')

    def segment(self):
        """
        Run inference
        """

        print('Computing probability maps...')

        self.model.eval()

        tbar = tqdm(self.testLoader, desc='\r')
        numImgs = len(self.testLoader.dataset)

        fileList = []

        for i, sample in enumerate(tbar):

            image = sample['image']
            imgSiz = sample['imagesize']
            height = imgSiz[0][0]
            width = imgSiz[1][0]
            fileList.extend(sample['fname'])

            if self.cuda:
                image = image.cuda()
            with torch.no_grad():
                output = self.model(image)

            # Get probability maps
            sm = torch.nn.Softmax(dim=1)
            prob = sm(output)

            # Reshape probability map
            probMap = np.squeeze(prob.cpu().numpy())

            if i == 0:
                probMapAll = np.zeros((self.nClass, height, width, numImgs))

            for c in range(0, self.nClass):
                classProbMap = probMap[c, :, :]
                resizProbMap = resize(classProbMap, (height, width), anti_aliasing=True)
                probMapAll[c, :, :, i] = resizProbMap

        if self.view == 'axial':
            probMapAllAx = probMapAll
        elif self.view == 'sagittal':
            # Transform to axial orientation
            probMapAllAx = np.transpose(probMapAll, (0, 2, 3, 1))
        elif self.view == 'coronal':
            # Transform to axial orientation
            probMapAllAx = np.transpose(probMapAll, (0, 3, 2, 1))

        return probMapAllAx, fileList[-1]

def main(inputPath, outputPath):
    """
    Run model ensemble and fuse results
    """

    # Get probability maps from different views
    print('Beginning inference...')
    t0 = process_time()
    larynxAx = Larynx(inputPath, 'axial')
    probMapAx, fName = larynxAx.segment()
    nClass = larynxAx.nClass
    print("%.1f" % (process_time() - t0) + ' s')


    t1 = process_time()
    larynxSag = Larynx(inputPath, 'sagittal')
    probMapSag, __ = larynxSag.segment()
    print("%.1f" % (process_time() - t1) + ' s')


    t2 = process_time()
    larynxCor = Larynx(inputPath, 'coronal')
    probMapCor, __ = larynxCor.segment()
    print("%.1f" % (process_time() - t2) + ' s')


    # Fuse maps
    print('Generating consensus segmentations...')
    t3 = process_time()
    avgProb = (probMapAx + probMapSag + probMapCor) / 3
    labels = np.argmax(avgProb, axis=0)
    mask = labelToBin(labels, nClass-1)
    print("%.1f" % (process_time() - t3) + ' s')


    # Write to NIfTI
    print('Writing output mask to disk...')
    t4 = process_time()
    #path, file = os.path.split(fName)
    inputImg = sitk.ReadImage(fName)
    writeFile(mask, outputPath, inputImg)

    print("%.1f" % (process_time() - t4) + ' s')

    return mask

if __name__ == '__main__':
    if 'MKL_NUM_THREADS' in os.environ:
        del os.environ['MKL_NUM_THREADS']
    main(sys.argv[1], sys.argv[2])
