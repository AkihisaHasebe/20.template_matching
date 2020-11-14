import xml.etree.ElementTree as ET
import cv2


tree = ET.parse('../img/lize_00811_0.xml')

root = tree.getroot()

crop_pos = []

for child in root.find('object').find('bndbox'):
    crop_pos.append(int(child.text))

img = cv2.imread('../img/lize_00811_0.png')

roi = img[crop_pos[1]:crop_pos[3],crop_pos[0]:crop_pos[2]]

cv2.imwrite('../img/template.png',roi)