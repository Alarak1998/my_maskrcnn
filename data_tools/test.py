import cv2
import numpy as np

# 读取图像
img = cv2.imread(r'D:\anaconda3\envs\pytorch\mask_rcnn_pytorch\data_tools\000000000016.jpg')

# 定义第一个多边形的顶点坐标
points1 = np.array([[160, 130], [350, 130], [250, 300]])

# 定义第二个多边形的顶点坐标
points2 = np.array([[60, 30], [150, 30], [100, 100]])

# 将所有的多边形顶点坐标放入一个列表中
points = [points1, points2]

print(points)
# 使用cv2.fillPoly()函数来填充多边形
mask = cv2.fillPoly(img, pts=points, color=(255, 0, 0))

# 显示图像
# cv2.imshow('Polygons', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('_mask.png', mask)