IMPORT PYTHON3 AS PYTHON;
IMPORT Std;
IMPORT GNN.Tensor;
IMPORT $.Types;

t_Tensor := Tensor.R4.t_Tensor; 

EXPORT ImageConverter := MODULE

   // Overview:
    // This module tends towards the input, output and manipulation of images pertaining to neural network applications.  
    // This makes sure that the users of GNN do not spend time trying to preprocess the image database, 
    // as the images to be processed are read and generates the corresponding ECL tensors. 
    
    // The module is capable of taking datasets of images sprayed as a blob, usually with the prefix: [filename,filesize]
    // This dataset sprayed is taken to obtain the image matrix so as to be sent to the neural network. 
    // This module handles the preprocessing. It can convert records containing images as byte data into Tensor data 
    //to be able to use for conversion into a tensor and train the neural network using the tensor.  
    //ACTIVITY is used to run the python program in all the nodes
    

    EXPORT STREAMED DATASET(t_Tensor) pyConvertImages(STREAMED DATASET(Types.ImgRec) imgs) := EMBED(Python:activity)
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import io
        import math
        global Np2Tens
        #S Function to convert a NumPy array to t_Tensor and yield the tensor slices
        def _Np2Tens(a, wi=0, maxSliceOverride=0, isWeights = False):
            #   dTypeDict is used to convey the data type of a tensor.  It must be
            #   kept in sync with the Tensor data types in Tensor.ecl
            #S  Data type dictionary to map tensor data types
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
            datType = 1         # Set for default data type (float32), on other case use the declaration just above
            elemSize = 4        # Set for default element size (4 bytes), on other case use the declaration just above
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
        #this is a generator function to iterate through multiple images
        #generator function is needed as Np2Tens doesn't return anything but yeilds.
        #the np.frombuffer interprets a buffer as a 1-dimensional array(here interpreting whole image data into 1d array).
        #plt.imread is used to decode the image,(i.e reads an image content using ByteIO function and returns the numpy.array).
        #the imread() function is passed an option parameter('ext' in this case), which takes in the image format, format is extarcted by spliting the filename with '.'.
        #also, Np2Tens function takes care of ndarray to float values, thats why not explicitly not converting the ndarray to float value
        #here in the inner for-loop, we are calling the Np2tens function with optional parameter (wi=id, which is setting workitem to image id) which yields the tensors.
        def generateTensors(imageRecs):
            for rec in imageRecs:
                id,filename,img = rec
                tokens = filename.split('.')
                ext = tokens[1].lower()
                image_np = np.frombuffer(img, dtype='uint8')
                image = plt.imread(io.BytesIO(image_np), ext) #uncompresses the image data
                for ten in Np2Tens(image,wi=id):
                    yield ten
        
        #calling the generator function.
        try:
            return generateTensors(imgs)
        #The format_exc() function from the traceback module is called to retrieve the formatted traceback of the exception that occurred.
        #The traceback string is stored in the variable exc.
        #An assert statement is used with False as the condition and exc as the error message. This line effectively raises an assertion error, 
        #with the traceback string as the error message.
        except:
            import traceback as tb
            exc = tb.format_exc()
            assert False, exc
    ENDEMBED;
    
    
    //This fuction will Distribute the images received in ImgRec format equally to all nodes
    //with multiple nodes running, timing of execution can be different for each, therefore tensors_s is declared with SORT,
    //so that to put back the slices returned to its canonical order.
    EXPORT DATASET(t_Tensor) convertImages(DATASET(Types.ImgRec) images) := FUNCTION
        imagesD := DISTRIBUTE(images,id); //distributes the data to all nodes in a modular fasion, using id as the ditribution key.
        tensors := pyConvertImages(imagesD);
        tensors_s := SORT(tensors,wi,sliceId);
        return tensors_s;

    END;



END; //end of module
