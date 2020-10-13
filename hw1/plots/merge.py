from PIL import Image
import numpy as np
import os
img_list = [file for file in os.listdir("./") if ".png" in file]
print(img_list)
for i in range(0, len(img_list), 2):
    img1 = Image.open(img_list[i])
    img2 = Image.open(img_list[i+1])
    width, height = img1.size
    result = Image.new(img1.mode, (width*2, height))
    result.paste(img1, box=(0, 0))
    result.paste(img2, box=(width, 0))
    result.save("results/"+img_list[i][:-8]+".png")
