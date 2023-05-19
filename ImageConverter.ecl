IMPORT PYTHON3 AS PYTHON;
IMPORT Std;
IMPORT GNN.Tensor;
IMPORT $.Types;

t_Tensor := Tensor.R4.t_Tensor;


EXPORT ImageConverter := MODULE


    EXPORT STREAMED DATASET(t_Tensor) pyConvertImages(STREAMED DATASET(Types.ImgRec) imgs) := EMBED(Python:activity)
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import io
        import math
        global Np2Tens
        # Returns a streamed dataset of t_Tensor
        def _Np2Tens(a, wi=0, maxSliceOverride=0, isWeights = False):
            #   dTypeDict is used to convey the data type of a tensor.  It must be
            #   kept in sync with the Tensor data types in Tensor.ecl
            dTypeDict = {1:np.float32, 2:np.float64, 3:np.int32, 4:np.int64}
            dTypeDictR = {'float32':1, 'float64':2, 'int32':3, 'int64':4}
            #   Store the element size for each tensor data type.
            dTypeSizeDict = {1:4, 2:8, 3:4, 4:8}
            maxSliceLen = 1000000
            nNodes = 1
            nodeId = 0
            epsilon = .000000001
            origShape = list(a.shape)
            flatA = a.reshape(-1)
            flatSize = flatA.shape[0]
            currSlice = 1
            indx = 0
            #datType = dTypeDictR[str(a.dtype)]
            #elemSize = dTypeSizeDict[datType]

            datType = 1
            elemSize = 4
            if maxSliceOverride:
                maxSliceSize = maxSliceOverride
            else:
                maxSliceSize = divmod(maxSliceLen, elemSize)[0]
            if isWeights and nNodes > 1 and flatSize > nNodes:
                # When we are synchronizing weights, we need to make sure
                # that we create Tensor with at least 1 slice per node.
                # This allows all nodes to participate equally in the
                # aggregation of weight changes.  For other data, it
                # is more efficient to return fewer slices.
                altSliceSize = math.ceil(flatSize / nNodes)
                maxSliceSize = min([maxSliceSize, altSliceSize])
            while indx < flatSize:
                remaining = flatSize - indx
                if remaining >= maxSliceSize:
                    sliceSize = maxSliceSize
                else:
                    sliceSize = remaining
                dat = list(flatA[indx:indx + sliceSize])
                dat = [float(d) for d in dat]
                elemCount = 0
                for i in range(len(dat)):
                    if abs(dat[i]) > epsilon:
                        elemCount += 1
                if elemCount > 0 or currSlice == 1:
                    if elemCount * (elemSize + 4) < len(dat):
                        # Sparse encoding
                        sparse = []
                        for i in range(len(dat)):
                            if abs(dat[i]) > epsilon:
                                sparse.append((i, dat[i]))
                        yield (nodeId, wi, currSlice, origShape, datType, maxSliceSize, sliceSize, [], sparse)
                    else:
                        # Dense encoding
                        yield (nodeId, wi, currSlice, origShape, datType, maxSliceSize, sliceSize, dat, [])
                currSlice += 1
                indx += sliceSize

        Np2Tens = _Np2Tens
        def generateTensors(imageRecs):
            for rec in imageRecs:
                id,filename,img = rec
                tokens = filename.split('.')
                ext = tokens[1].lower()
                image_np = np.frombuffer(img, dtype='uint8')
                image = plt.imread(io.BytesIO(image_np), ext)
                for ten in Np2Tens(image,wi=id):
                    yield ten

        try:
            return generateTensors(imgs)
        except:
            import traceback as tb
            exc = tb.format_exc()
            assert False, exc
    ENDEMBED;

    EXPORT DATASET(t_Tensor) convertImages(DATASET(Types.ImgRec) images) := FUNCTION
        imagesD := DISTRIBUTE(images,id);
        tensors := pyConvertImages(imagesD);
        tensors_s := SORT(tensors,wi,sliceId);
        return tensors_s;

    END;



END;