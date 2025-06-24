import numpy as np
import cv2

rgb=[(241, 143, 1),(4, 139, 168), (247, 243, 227),(153, 93, 129),(222, 193, 255), (0, 5, 5), (109, 163, 77) ]
def hex_to_rgb(colors):
    hex_colors = c.split()

    for hex in hex_colors:
        r, g, b = tuple(int(hex[i:i+2], 16) for i in (1,3,5))
        bgr = (b,g,r)
        print(f"HEX {hex} converted to bgr {bgr}")

c = """#FFF07C
#FFF07C
#F7F3E3
#995D81
#99C24D
#DEC1FF
#a65628
#f781bf
#999999"""

print(hex_to_rgb(c))
    
    