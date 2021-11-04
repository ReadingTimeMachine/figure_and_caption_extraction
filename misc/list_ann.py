from glob import glob
import numpy as np
import pandas as pd

# resave
wspageList = '/Users/jillnaiman/Downloads/tmp/presavelist.txt'

fann = '/Users/jillnaiman/tmpMegaYolo/binaries/yolo_512x512_cap_math_ann/*xml'

pdfDir = '/Users/jillnaiman/tmpADSDownloads/pdfs/'


flist = glob(fann)

ws,ps = [],[]
for l in flist:
    w=l.split('/')[-1].split('_p')[0]
    p=l.split('_p')[-1].split('.xml')[0]
    ws.append(pdfDir+w+'.pdf'); ps.append(int(p))

df = pd.DataFrame({'filename':ws, 'pageNum':ps})

df.to_csv(wspageList,index=False)
