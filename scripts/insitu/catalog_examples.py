"""Offline preparation only: write complete records, never inherit at runtime."""
import json
from math import pi
from pathlib import Path


def tgv32_products(interval=2,statistics=True):
    example=Path(__file__).resolve().parent.parent/'output/input.products.tgv.images.example.jsonl'
    records=[json.loads(line) for line in example.read_text().splitlines()][1:]
    records=[row for row in records if statistics or row['source']=='instantaneous']
    for row in records: row['schedule']=dict(mode='steps',interval=interval)
    return records


def tgv256_products(interval=1,write_eps=False):
    records=[row for row in tgv32_products(interval,statistics=False) if row['kind'] in ('q_surface','streamlines')]
    for row in records:
        row['id']='q_surface' if row['kind']=='q_surface' else 'instantaneous_streamlines'
        row['field']='velocity_magnitude'
        row['display'].update(camera=[13.,11.,12.],focal_point=[pi]*3,view_up=[0.,0.,1.],
            parallel_scale=5.,color_range=[0.,1.],lighting=True)
        row['output'].update(image_size=[1280,960],formats=['jpeg','eps'] if write_eps else ['jpeg'])
        if row['kind']=='q_surface': row['extraction']['isovalue']=0.
        else:
            h=2*pi/256
            row['display']['line_width']=3.
            row['extraction'].update(seed_layout='line16',initial_step=.1*h,minimum_step=.01*h,
                maximum_step=.5*h,maximum_length=pi)
    return records


def m12_short_products(interval=5):
    """The already validated 64x32x24 interface fixture, not the production grid."""
    example=Path(__file__).resolve().parent.parent/'output/input.products.m12.short.example.jsonl'
    records=[json.loads(line) for line in example.read_text().splitlines()][1:]
    for row in records: row['schedule']=dict(mode='steps',interval=interval)
    return records


def write_catalog(path,records):
    Path(path).write_text('\n'.join(json.dumps(row,allow_nan=False) for row in
        [dict(catalog_schema='astr-insitu-products-v1'),*records])+'\n')
