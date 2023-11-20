import json
import cv2
import os
import numpy as np

 
file_path = './data/m_coco/train2017/'
save_path = './data/m_coco/train_mask/'

def generate_masks(args):
    for file_name in os.listdir(args.file_path):
        if file_name.endswith('.json'):
            tmp = {}
            with open(os.path.join(args.file_path, file_name), "r") as f:
                tmp = f.read()
            
            tmp = json.loads(tmp)
            poly_num = sum(isinstance(i, dict) for i in tmp["shapes"])
            
            # all_polys = []
            for i in range(poly_num):
                
                mask_name = tmp["shapes"][i]["label"]
                points = tmp["shapes"][i]["points"]
                points = np.array(points, np.int32)
            
                img = cv2.imread(args.file_path + file_name[0:-5] + '.tiff')
                #BGR->RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # box = tmp["shapes"][1]["points"]
                # box = np.array(box, np.int32)
                
                mask = np.zeros_like(img)
                
                # cv2.rectangle(img,(box[0][0], box[0][1]), (box[1][0], box[1][1]) ,(125,125,125),2)
                #cv2.polylines(img, [points], 1, (0,0,255))
                
                # all_polys.append(points)
            
                # print(all_polys)
                cv2.fillPoly(mask, [points], (255, 255, 255))
                # img_add = cv2.addWeighted(mask, 0.3, img, 0.7, 0)
                save_root = args.save_path + file_name[0:-5] + '_{}_{}.png'.format(mask_name, i)
                print(save_root)
                cv2.imwrite(save_root, mask)
                # plt.imshow(img_add)
                # plt.show()

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--file_path', default='./data/m_coco/train2017/')
    parser.add_argument('--save_path', default='./data/m_coco/train_mask/')
    
    args = parser.parse_args()
    print(args)

    generate_masks(args)