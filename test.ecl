IMPORT PYTHON3 AS PYTHON;
IMPORT Std;
IMPORT GNN.Tensor;
IMPORT ImageConverter;
IMPORT $.Types;



rawImageRec := RECORD
    STRING filename;
    DATA image;
  END;


ds := DATASET('poke1.flat', rawImageRec, THOR);
OUTPUT(ds);
ds2 := PROJECT(ds,TRANSFORM(Types.ImgRec,self.id:=COUNTER,SELF:=LEFT));

OUTPUT(ds2);

dat := ImageConverter.convertImages(ds2);
dat2 := PROJECT(dat,{UNSIGNED wi, UNSIGNED sliceId, SET OF UNSIGNED shape, UNSIGNED sliceSize, UNSIGNED maxSliceSize});


OUTPUT(dat2);
//OUTPUT(dat,{nodeId,wi,sliceId,shape,sliceSize});
//OUTPUT(dat);