output_file = './recorded.txt' 
mode = 1
mode_cand = ['add_file','add','overwrite']
# add_file will skip recorded file
# add will process each recorded file one more time
# overwrite will process each recorded file one more time
_m = mode_cand[mode]

from os.path import *
from glob import glob
from tqdm import tqdm
import cv2
from collections import defaultdict
if __name__ == '__main__':
    if  exists(output_file):
        recorded_f = [_.split('\t') 
                    for _ in open(output_file).read().strip().split('\n')]
        recorded_f = {k[0]:[(int(e.split(',')[0]),int(e.split(',')[1]))
                            for e in k[1:]] 
                    for k in recorded_f}
    else:
        recorded_f = {}
        
    #the [x, y] for each right-click event will be stored here
    right_clicks = defaultdict(list)
    #this function will be called whenever the mouse is right-clicked
    def mouse_callback(event, x, y, flags, params):
        #right-click event value is 2
        if event == 2:
            global right_clicks
            #store the coordinates of the right-click event
            right_clicks[name].append([x, y])
            #this just verifies that the mouse data is being collected
            #you probably want to remove this later
            #tqdm.write(f"{x}, {y}")

    name2point = {}
    all_f = tqdm(glob('./**/*.JPG'))
    for path_image in all_f:
        name = path_image.replace('.\\','')  # .\\ for windows path
        if _m in ['add_file'] and name in recorded_f:
            # skip recorded file
            continue
        elif _m in ['add','overwrite']:
            # process it 
            if len(recorded_f.get(name,[]))>=3:continue
            pass
        img = cv2.imread(path_image)
        scale_width = 640 / img.shape[1]
        scale_height = 480 / img.shape[0]
        scale = min(scale_width, scale_height)
        window_width = int(img.shape[1] * scale)
        window_height = int(img.shape[0] * scale)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', window_width, window_height)
        #set mouse callback function for window
        cv2.setMouseCallback('image', mouse_callback, name)
        cv2.imshow('image', img)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
        
    if _m in ['add_file']:
        # update it directly
        right_clicks.update(recorded_f)
    elif _m in ['add']:
        for k,v in recorded_f.items():
            right_clicks[k] += v
    elif _m in ['overwrite']:
        # overwrite it
        pass
    with open(output_file,'w') as f1:
        for k,v in right_clicks.items():
            f1.write(f"{k}\t" + '\t'.join([f"{_v[0]},{_v[1]}" for _v in v]) + '\n')
        