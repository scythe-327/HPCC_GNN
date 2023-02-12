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
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
######----------------------------------------- Contents in 03_GNN_Project_Backup.ipynb  ------------------------------------------------------------------------------------


# -This note book is the iterartion 03 of the HPCC_GNN_Project-
# 1.Struct.unpack  is used to strip the headers, in the JPEG images.
# 2.A manual function is written therfore ensures a higher control over the data
# 3.Assumes that the input jpeg_data has already been decoded into raw image data, which may not be the case for all JPEG images.
import io
from PIL import Image
import binascii
import numpy as np
from io import BytesIO


# Since the blob spraying returns the hex data of an image , we are trying to simulate the same by using io and Pil library. (In this note book we have used a png format)
img=Image.open("C:/Users/rohan/Desktop/uktkjxhcb/sampleFormats/902587.jpg")
print(img)



# Uing hexlify() imported from binascii library to obtain the hex data of the image.
with open("C:/Users/rohan/Desktop/uktkjxhcb/sampleFormats/902587.jpg", "rb") as image_file:
    image_hex = binascii.hexlify(image_file.read())
    # print(image_he
    print(image_hex[:1500])
    
    
#  The hex data into binary data using unhexlify() function.   
binary_data = binascii.unhexlify(image_hex)



# You can access the height and width of a JPEG image using 
def get_jpeg_dimensions(jpeg_data):
    # Open the JPEG data as an image using the Pillow library
    image = Image.open(io.BytesIO(jpeg_data))

    # Get the width and height of the image
    width, height = image.size

    return width, height
width,height=get_jpeg_dimensions(binary_data)
print(width,height)




# Extractring pixel data


def decode_jpeg(jpeg_data):
    # Open the JPEG data as an image using the Pillow library
    image = Image.open(io.BytesIO(jpeg_data))

    # Access the raw image data as a binary string
    raw_image_data = image.tobytes()

    return raw_image_data
def convert_to_rgb(raw_image_data):
    # Convert the raw image data into a format suitable for processing
    image = Image.frombytes("RGB", (width, height), raw_image_data)

    # Access the image data as a binary string
    image_data = image.tobytes()

    return image_data
def extract_pixel_data(jpeg_data):
    # Decode the JPEG data
    raw_image_data = decode_jpeg(jpeg_data)

    # Convert the raw image data into a format suitable for processing
    image_data = convert_to_rgb(raw_image_data)

    # Access the pixel data
    pixels = []
    for i in range(0, len(image_data), 3):
        # Unpack the next three bytes as three 8-bit values representing red, green, and blue
        r, g, b = struct.unpack("BBB", image_data[i:i+3])
        pixels.append((r, g, b))

    return pixels
jpeg_data = binary_data
print(extract_pixel_data(jpeg_data))
# thus it is easy to convert the binary pixel data obtained into tensors via nparray

####----------------------------------------------------------- EOF --------------------------------------------------------------------------------------------------------
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
######----------------------------------------- Contents in 04_GNN_Project_Backup.ipynb  ------------------------------------------------------------------------------------



# -This note book is the iterartion 04 of the HPCC_GNN_Project-

# References:https://www.ece.ualberta.ca/~elliott/ee552/studentAppNotes/2003_w/misc/bmp_file_format/bmp_file_format.htm
# 1.Image type being processed is BMP
# 2.A manual function using struct.unpack is written which ensures a higher control over the data
# 3.The main aim is to extract the pixel data from the hexdata generated from spraying the data.




import io
import struct
from PIL import Image
import binascii
import numpy as np
from io import BytesIO

# Since the blob spraying returns the hex data of an image , we are trying to simulate the same by using io and Pil library.
# (In this note book we have used a BMP format)
# img=Image.open("C:/Users/rohan/Pictures/Camera Roll/X13_wallpaper_final_16x9_FHD.jpg")
img=Image.open("C:/Users/rohan/Desktop/uktkjxhcb/sampleFormats/sample_1920x1280_BMP.bmp")
print(img)


# Uing hexlify() imported from binascii library to obtain the hex data of the image.
with open("C:/Users/rohan/Desktop/uktkjxhcb/sampleFormats/sample_1920x1280_BMP.bmp", "rb") as image_file:
    image_hex = binascii.hexlify(image_file.read())
    # print(image_he
    print(image_hex[:1500])
 

# The hex data into binary data using unhexlify() function.
binary_data = binascii.unhexlify(image_hex)


# struct.unpack returns a tuple, even if it contains only one value
# And hence we need to compare it individually
# The BMP format uses the two-byte sequence 0x42 0x4D


def is_bitmap(image_hex):
    bmp_signature = b'BM'
    unpacked_signature = struct.unpack("2B", binary_data[:2])
    return bytes([unpacked_signature[0]]) == bmp_signature[:1] and bytes([unpacked_signature[1]]) == bmp_signature[1:]
print(is_bitmap(image_hex))


# Pixels are stored "upside-down" with respect to normal image raster scan order, starting in the lower left corner, going from left to right,
# and then row by row from the bottom to the top of the image.2 Uncompressed Windows bitmaps can also be stored from the top row to the bottom, 
# if the image height value is negative.(https://grapherhelp.goldensoftware.com/subsys/subsys_bitmap_file_description.htm)
# struct unpack is used to extract these header in the form of "<IHHIIIIIIII "
def extract_pixel_data(image_hex):
    # Unpack the header information from the binary data
    header = struct.unpack("<IHHIIIIIIII", image_hex[:40])
    image_width = header[6]
    image_height = header[7]
    pixel_offset = header[10]
# Below we are calculating the pixel data size
# Since the BMP image must be a multiple of 4 so it will be just rounded to the nearest 4's multiple (https://stackoverflow.com/questions/15313792/what-should-we-do-with-a-bitmap-file-when-its-rowsize-isnt-multiple-of-4)
    row_size = (image_width * 3 + 3) & ~3
    image_size = row_size * abs(image_height)


    pixel_data = image_hex[pixel_offset:pixel_offset + image_size]

    return pixel_data
pixel_data = extract_pixel_data(binary_data)
# printing the first 1000 indexes of  pixel data for verification purposes
print(pixel_data[:1000])
