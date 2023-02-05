# Our idea is to obtain the whole data of an image and to play around a bit to see what can be done using python libraries 
# then to embed the desired code blocks to ecl
# the below codes conatins python snippets we tried
# for better understanding cases why we used any of the code snippetes below pls refer the 3 notebooks in this repo :)


import io
from PIL import Image
import binascii
import pickle
import tensorflow as tf
from io import BytesIO
import numpy as np
import struct

######----------------------------------------- Contents in 00_GNN_Project_Backup.ipynb  ------------------------------------------------------------------------------------




#open the image 
img=Image.open("--Imagename--")
print(img)

#Here the image must be opened in read bytes format
with open("--Imagepath--", "rb") as image_file:
    image_hex = binascii.hexlify(image_file.read())
    print(image_hex[:1500])

# converting the hex data into binary data using unhexlify() function.
binary_data = binascii.unhexlify(image_hex)
print(binary_data)
with open('--file_which_contains_binarydata_of_image--', 'wb') as f:
    pickle.dump(binary_data, f)
f.close();
# the above snippet returns the string of binary data

# function to interpret the binary data as an image and return the image in the form of a tensor.

image_tensor = tf.io.decode_raw(binary_data,out_type=tf.uint8)
print(image_tensor)  #prints the tensor data

#Please keep in mind that the pickle and json methods will not be able to save the tensorflow tensor object, rather it will save the numpy array representation of the tensor.

with open('--file_which_contains_tensordata_of_image--', 'wb') as f:
    pickle.dump(image_tensor, f)
f.close();




with open('--file_which_contains_tensordata_of_image--', 'rb') as f:
    # Load the tensor from the file
    image_tensor = pickle.load(f)
    print(image_tensor.shape)


####----------------------------------------------------------- EOF --------------------------------------------------------------------------------------------------------

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

######----------------------------------------- Contents in 01_GNN_Project_Backup.ipynb  ------------------------------------------------------------------------------------


##-
## 1.Inorder to strip the headers we are utilizing the Pillow library.
## 2.As per suggestions given in the previous meet the byte data is now converted to ndarray using the Numpy library instead of tensrflow.

# will open the image as in the above code using binascii library(discussed in line 21 t0 31)


# To check if the image data is png we check the first 16 bits
png_signature = "89504e470d0a1a0a"
if image_hex[:16] == png_signature:
    image_hex = image_hex[16:]

#The method strip_headers(binary_data) using the Pillow library strips the headers from the PNG image.
#The Image.open() method reads the binary data as an image and automatically strips the headers and metadata to give you a representation of the image data.
#The image data is then converted to a NumPy array using np.array(). 
#This gives you a NumPy array that only contains the image pixel data, with the headers and metadata stripped.

def strip_headers(binary_data):
    with Image.open(BytesIO(binary_data)) as im:
        return np.array(im)

stripped_image_array1 = strip_headers(binary_data)
print(stripped_image_array1)

# function to interpret the binary data as an image and return the image in the form of a tensor.
image_tensor =tf.convert_to_tensor(stripped_image_array1)
print(image_tensor) 

#REVERTING THE TENSOR BACK TO THE IMAGE
np_ar = image_tensor.numpy()
print(np_ar)

# nd arrray to bytes
bts = np_ar.tobytes()
print(bts)   ## prints really long string of bytes

# bytes to hex
res = binascii.hexlify(bytearray(bts))
print(res[:1900])

####----------------------------------------------------------- EOF --------------------------------------------------------------------------------------------------------

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

######----------------------------------------- Contents in 02_GNN_Project_Backup.ipynb  ------------------------------------------------------------------------------------

#-
#1.Struct.unpack  is used to strip the headers, this can be done by extracting the pixel data present in the IDTA chunk of the png image.
#2.A manual function is written therfore ensures a higher control over the data

#Since the blob spraying returns the hex data of an image , we are trying to simulate the same by using io and Pil library. (In this note book we have used a png format)
# will open the image as in the above code using binascii library(discussed in line 21 t0 31)

#The hex data into binary data using unhexlify() function.
binary_data = binascii.unhexlify(image_hex)
#
signature = binary_data[0:8]
if signature != b'\x89PNG\r\n\x1a\n':
    raise ValueError("Invalid PNG data")

#Search for the IHDR chunk
index = 8     #will start from the next index of signature
while index < len(binary_data):
    # Extract the chunk length,
    #  After the chunk length is extracted using struct.unpack, the chunk type is extracted using slicing: chunk_type = binary_data[index+4:index+8].
    chunk_length, = struct.unpack("!I", binary_data[index:index+4])
    chunk_type = binary_data[index+4:index+8]
    if chunk_type == b'IHDR':
        break
    else:
        # Skip over this chunk
        index += chunk_length + 12
print(chunk_length,"\t",chunk_type,"\t",index)    # this will return the length of chunk, tyoe of the chunk and the starting index of chunk

#Extract the image dimensions from the IHDR chunk.
width, height = struct.unpack("!II", binary_data[index+8:index+16])
print(width,"\t",height)    #retuns width and height in bits

#Once the chunk type is found to be IDAT, we can extract the actual image data from the binary data. This is done by slicing the binary data from the position index + 8 to 
#the position index + 8 + chunk_length. The reason we start from index + 8 is because the PNG chunk structure consists of:
#4 bytes for the chunk length
#4 bytes for the chunk type
#chunk_length bytes for the chunk data
#4 bytes for the chunk crc   
#So the actual image data starts at the position index + 8, which is 4 bytes after the chunk length and 4 bytes after the chunk type. The end position is index + 8 + chunk_length, 
# which is chunk_length bytes after the start position.

# loop through the dat till we find the IDAT chunk (a chunk where the actual pixel image data start)
while index < len(binary_data):
    chunk_length, = struct.unpack('!I', binary_data[index:index+4])
    chunk_type = binary_data[index+4:index+8]
    if chunk_type == b'IDAT':
        # Found the IDAT chunk
        image_data = binary_data[index+8:index+8+chunk_length]
        break
    index += chunk_length + 12
print(image_data)


##will it work?
def strip_headers(image_data):
    with Image.open(BytesIO(image_data)) as im:
        return np.array(im)
stripped_image_array1 = strip_headers(image_data)
image = np.frombuffer(image_data, np.uint8)
print(image)

# from the output of this snippet we can see since the header of the data has been stripped,there is an error thrown that it cant identify the image file


####----------------------------------------------------------- EOF --------------------------------------------------------------------------------------------------------


