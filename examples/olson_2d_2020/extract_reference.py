#!/usr/bin/env python3
"""Extract actual reference marker centers from Fig 6 vector PDF (page 15).
Usage: python extract_reference.py /path/to/2303.06634v1.pdf
Source: https://arxiv.org/pdf/2303.06634v1 (Steinberg & Heizler, 2023).
The open circles are Olson PN reference data; the lines are MC solutions.
Requires PyMuPDF. No simulated values enter this extraction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import fitz
import numpy as np

p=argparse.ArgumentParser(description=__doc__);p.add_argument('pdf',type=Path);args=p.parse_args()
out=Path(__file__).parent/'reference';out.mkdir(exist_ok=True)
page=fitz.open(args.pdf)[14]
colors={(0.,0.,0.):2.,(0.,0.,1.):2.5,(1.,0.,0.):3.}
rows=[]
for d in page.get_drawings():
    r=d['rect'];color=d['fill']
    # Circle paths have 8 Bezier segments, diameter ~2 PDF points.
    if d['type']!='f' or len(d['items'])!=8 or color not in colors:continue
    if not (1.9<r.width<2.2 and 1.9<r.height<2.2):continue
    x,y=(r.x0+r.x1)/2,(r.y0+r.y1)/2
    if not (518<y<654):continue
    ct=colors[color]
    if 124<x<304:
        # Exclude material legend markers (x ~252, y<597).
        if x>245 and y<600:continue
        # Frame: x=[0,1.8], log10(T_keV^4)=[-8,-3.8].
        radius=(x-124.0689926)/(302.9379883-124.0689926)*1.8
        logval=-8+(652.6229858-y)/(652.6229858-518.4709473)*4.2
        quantity='material_T4'
    elif 339<x<519:
        # Radiation legend samples are at x=356.908.
        if abs(x-356.908)<.02:continue
        # Frame: r=[0,2.25], log10(E/aTkeV^4)=[-8,-1.25].
        radius=(x-339.3120117)/(517.7210083-339.3120117)*2.25
        logval=-8+(652.8060303-y)/(652.8060303-518.9990234)*6.75
        quantity='radiation'
    else:continue
    rows.append((quantity,ct,radius,logval))
for quantity in ('material_T4','radiation'):
    selected=sorted((ct,r,v) for q,ct,r,v in rows if q==quantity)
    np.savetxt(out/f'{quantity}.csv',selected,delimiter=',',header='ct,r_cm,log10_value',comments='',fmt='%.10g')
    print(quantity,len(selected))
(out/'provenance.json').write_text(json.dumps({'source':'https://arxiv.org/pdf/2303.06634v1','figure':'6','page':15,'sha256':hashlib.sha256(args.pdf.read_bytes()).hexdigest(),'method':'PN reference-circle centers with axis-affine mapping; additional IMC material curves stored separately from dashed vector paths','coordinate_uncertainty_cm':0.01,'log10_uncertainty':0.03},indent=2)+'\n')

# Also preserve the published IMC material curves. These dashed paths are
# distinct from the PN open-circle markers and are never substituted for them.
imc=[]
for d in page.get_drawings():
    rect=d['rect']
    if d['type']!='s' or d['color'] not in colors:continue
    if not (rect.x0>124 and rect.x1<304 and rect.y0>518 and len(d['items'])>50):continue
    if not d['dashes'].startswith('[ 1.70895 '):continue
    for segment in d['items']:
        assert segment[0]=='l', 'Unexpected non-linear IMC path'
        for point in segment[1:]:
            radius=(point.x-124.0689926)/(302.9379883-124.0689926)*1.8
            logval=-8+(652.6229858-point.y)/(652.6229858-518.4709473)*4.2
            imc.append((colors[d['color']],radius,logval))
imc=sorted(set(imc))
assert len(imc)>300,'Missing published IMC curves'
np.savetxt(out/'published_imc_material_T4.csv',imc,delimiter=',',header='ct,r_cm,log10_value',comments='',fmt='%.10g')
print('Published IMC material',len(imc))
