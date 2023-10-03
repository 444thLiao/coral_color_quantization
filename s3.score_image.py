

# https://en.wikipedia.org/wiki/Color_quantization
"""
batch iterate cropped image and calculate it to the predefined color degree.

"""

from scipy.spatial.distance import cdist
import numpy as np
from PIL import Image
from collections import Counter


degree = {'D2':(201, 157, 96),
            'D3':(183, 97, 10),
            'D4':(143, 64, 27),
            'D5':(83, 43, 17),
            'D6':(61, 35, 18),}
    
    
def get_degree(f):
    image = Image.open(f).convert('RGBA')
    width, height = image.size
    pixels = image.getdata()
    pixel_count = width * height
    valid_pixels = []
    for i in range(0, pixel_count, 1):
        r, g, b, a = pixels[i]
        # If pixel is mostly opaque and not white
        if a >= 125:
            if not (r > 250 and g > 250 and b > 250):
                valid_pixels.append((r, g, b))

    a = np.array(valid_pixels)
    b = np.array(list(degree.values()))
    d = cdist(a,b)
    d_count = Counter([list(degree)[_] for _ in d.argmin(1)])
    cc = []
    for k,v in sorted(d_count.items()):
        #print(k,v/d.shape[0],)
        cc.append(int(k[1])*v/d.shape[0])
    num_d = sum(cc)
    return list(degree)[np.argmin(d.mean(0))],num_d


if __name__ == '__main__':
    from glob import glob
    from tqdm import tqdm
    cropped_ = sorted(glob('./found/**/*.JPG'))
    name2degree = {}
    name2num_degree = {}
    for i in tqdm(cropped_):
        deg,num_d = get_degree(i)
        name2degree[i] = deg
        name2num_degree[i] = num_d
        tqdm.write(f"{i}\t{deg}\t{num_d}")
        
    with open('./estimated_degree.tsv','w') as f1:
        f1.write('group1\tgroup2\tnumerical degree\n')     
        for i, v in name2num_degree.items():
            a,b = i.split('/')[-2:]
            f1.write(f"{a}\t{b}\t{round(v,4)}\n")

    # for _,row in df.iterrows():
    #     df.loc[_,'day'] = row['group1'].split('_')[-1]
    #     v = row['group2']
    #     if 'updated' not in v and 'upated' not in v:
    #         t = v.replace('.JPG','').split('_')[-1]
    #         df.loc[_,'treatment'] = d.get(t,t).upper()
    #         if t.startswith('DSC'):
    #             df.loc[_,'treatment'] = d[t+'.JPG'].upper()
    #     else:
    #         df.loc[_,'treatment'] = v.replace('.JPG','').split('_')[-2]
    # df.loc[:,'GROUP'] = [_[:-2] + 'P' if _[-1]=='P' else _[:-1] for _ in df['treatment']]
    # df.loc[:,'day(num)'] = [int(x.replace('day','')) for x in df['day']]
    # df = df.sort_values('day(num)')


    # import plotly.graph_objects as go
    # import plotly.express as px
    # import pandas as pd

    # group2color = {'LD':"#03a6c8",  'HD':"#077393",
    #             'LDP':"#F06292", 'HDP':"#C2185B",
    #             'CTRL':"#e2e2e2", 
    #             'LN':"#badf7c",'HN':"#4f8d57", 
    #             'LNP':"#BA68C8", 'HNP':"#4A148C" }
    # fig = px.box(df,y='numerical degree',color='GROUP',x='GROUP',facet_col='day')
    # fig.update_traces(boxpoints='all')
    # fig.write_html('./test.html')
    # fig.write_image('./test.pdf')