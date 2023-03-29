IMPORT PYTHON3 AS PYTHON;
IMPORT Std;

// Note that the filesize doesn't really show up anywhere, even though we spreayed with FILENAME,FILESIZE.
// It is embedded in the length of the contents field.
myRec := RECORD
  STRING filename;
  DATA contents;
END;

// I had sprayed a list of files (2) to 'thor::img.flat'.  Make sure you run both the spray and the job on the thor cluster.
ds := DATASET('img.flat', myRec, THOR);
// First output is just the file as we read it.
OUTPUT(ds);

// To show how to embed python, we just output the filename, length, and the first 16 bytes in hex.  This demonstrates
// that we were able to process the data.  Note the use of STREAMED DATASET and Python: Activity.
// This causes the python on each node to get all the records that happened to be on that node, and the resulting records
// are distributed the same as the input records to create a new distributed dataset.  This is how to do parallel processing
// in python. The curly braces form inside the output DATASET is just an inline record definition.  I could have
// declared another record type, which is a more normal way to do it.



DATASET({UNSIGNED4 len, STRING filename, STRING img_bits, STRING img_byte}) pyProcess(STREAMED DATASET(myRec) recs) := EMBED(Python: Activity)
  import binascii
  for rec in recs:
		# Records come in as a tuple of fields
    name, dat = rec
    header = dat[:16]
    img_hex = binascii.hexlify(dat)
    img_bytes=binascii.unhexlify(img_hex)
    
    headerstr = ''.join(format(x, '02x') for x in header)
    # We yield one record at a time (as a tuple), or we could have
    # made a list and returned the whole list as one.  Using yield,
    # we can process as a stream, without having to load everythin
    # into memory at the same time.
    yield (len(dat), name, img_hex, img_bytes)

ENDEMBED;


// Second output is the results returned by python
OUTPUT(pyProcess(ds))
