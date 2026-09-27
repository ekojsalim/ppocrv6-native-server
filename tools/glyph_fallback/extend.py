#!/usr/bin/env python3
"""Broaden evaluation across the CJK block, two held-out fonts and raster sizes.

Fonts and generated crops stay private. The production attachment is evaluation
only and is never used to generate a template. Defaults are deterministic.
"""
import argparse
import io
import json
import random
from pathlib import Path
from PIL import Image, ImageFilter
from prepare import digest, font_asset, render


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base-queries','droid','wenkai','vocabulary','production','output'):
        p.add_argument('--'+name, type=Path, required=True)
    args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    queries=json.loads(args.base_queries.read_text())
    vocab=set(args.vocabulary.read_text().splitlines())
    fonts=[]
    supported=[]
    for name,path in [('droid',args.droid),('wenkai',args.wenkai)]:
        _,coverage=font_asset(path,0,None,39)
        fonts.append((name,path));supported.append(coverage)
    coverage=supported[0]&supported[1]
    rng=random.Random(20260928)
    # New labels, not the first 1,024 development characters; balance model
    # vocabulary coverage so unknown labels do not disappear in an average.
    yes=[c for c in range(0x5200,0xA000) if c in coverage and chr(c) in vocab]
    no=[c for c in range(0x5200,0xA000) if c in coverage and chr(c) not in vocab]
    codes=sorted(set(rng.sample(yes,256)+rng.sample(no,256)) | set(map(ord,'岂岜岩未末土士口囗日曰己已巳人入八')))
    for name,path in fonts:
        for size,variant in [(39,'clean'),(29,'small_jpeg')]:
            font,_=font_asset(path,0,None,size)
            for code in codes:
                im=render(font,chr(code),size+18)
                if variant=='small_jpeg':
                    im=im.filter(ImageFilter.GaussianBlur(.35))
                    buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=80);buffer.seek(0)
                    im=Image.open(buffer).copy()
                ident=f'{name}_{variant}_{code:04X}'
                file=out/(ident+'.png');im.save(file)
                queries.append({'id':ident,'path':str(file),'group':f'{name}_{variant}_'+('iv' if chr(code) in vocab else 'oov'),
                                'expected':chr(code),'model':None})
    queries.append({'id':'production_attachment','path':str(args.production.resolve()),'group':'production_confirmed',
                    'expected':'岂','model':None})
    (out/'queries.json').write_text(json.dumps(queries,ensure_ascii=False))
    (out/'provenance.json').write_text(json.dumps({'seed':20260928,'fonts':[
        {'name':n,'path':str(f.resolve()),'sha256':digest(f),'dictionary_member':False} for n,f in fonts],
        'codes':codes,'production':{'sha256':digest(args.production),'label':'岂','label_source':'user confirmed'},
        'note':'No template generation or threshold tuning on these new labels.'},ensure_ascii=False,indent=2))
    print(len(queries),'queries',len(codes),'new labels')


if __name__=='__main__':
    main()
