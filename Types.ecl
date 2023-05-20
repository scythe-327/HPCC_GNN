EXPORT Types := MODULE
  
  //This module contains the Image Record, which consists of:-
  //  1. id: an id for each image sprayed,
  //  2. filename of the images,
  //  3. image: the contents of the image
  
  EXPORT ImgRec := RECORD
        UNSIGNED id;
        STRING filename;
        DATA image;
    END;
  
END;
