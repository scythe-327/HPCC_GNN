//This is a test file, where call to ImageConverter module and Types is declared
IMPORT PYTHON3 AS PYTHON;
IMPORT Std;
IMPORT GNN.Tensor;
IMPORT ImageConverter;
IMPORT $.Types;

// another Image RECORD, whith just the fuilename and image contentes, used for TRANSFORM of ImgRec
rawImageRec := RECORD
    STRING filename;
    DATA image;
  END;


ds := DATASET('poke1.flat', rawImageRec, THOR);
OUTPUT(ds);

//ds2 transforms ImgRec to rawImageRec RECORD,
//it sets the id to a counter for sequential id,
//SELF:=LEFT takes the input record, it basically sets all other attributes to input record (in this case ImgRec)
ds2 := PROJECT(ds,TRANSFORM(Types.ImgRec,self.id:=COUNTER,SELF:=LEFT));

OUTPUT(ds2);

dat := ImageConverter.convertImages(ds2);
dat2 := PROJECT(dat,{UNSIGNED wi, UNSIGNED sliceId, SET OF UNSIGNED shape, UNSIGNED sliceSize, UNSIGNED maxSliceSize});


OUTPUT(dat2);

