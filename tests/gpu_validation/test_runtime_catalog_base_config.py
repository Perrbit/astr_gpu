"""The native catalog is the only product-clock authority, including after restart."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from run_output_restart_validation import run_case,compare_fields
from test_runtime_catalog_images import arguments,dynamic_image,image_segments
from test_insitu_device_products import resident_configuration
from test_index_slice_runtime import ROOT

EXE=os.environ.get('ASTR_RUNTIME_BASE_CONFIG_EXE')
pytestmark=pytest.mark.skipif(not EXE,reason='Select the root native-catalog executable')


def test_existing_periodic_curve_plane_admission_does_not_depend_on_old_geometry(tmp_path):
    args=arguments(tmp_path);args.executable=Path(EXE).resolve()
    config=resident_configuration('standard-device',statistics=False,interval=1,profile='velocity_slice')
    config=config.replace('&insitu_run',"&insitu_run slice_definition='plane',slice_origin=0,0,1,slice_normal=0,0,1,")
    item=dynamic_image('physical_plane','plane')
    case,_=run_case(args,ROOT,'gpu',2,'periodic-curve',2,grid='32,32,32',tgv_mapping='periodic',
        insitu_config=config,product_catalog_records=[item])
    state=json.loads((case/'outdat/new/products.status.json').read_text())
    assert state['products'][0]['state']=='active',state
    assert len(list(image_segments(case)[0].glob('*.jpeg')))==2


@pytest.mark.parametrize('pipeline',['standard-device','compatible'])
def test_omitted_and_changed_old_clocks_do_not_change_catalog_continuation(tmp_path,pipeline):
    args=arguments(tmp_path);args.executable=Path(EXE).resolve();args.statistics=True
    config=resident_configuration(pipeline,statistics=True,interval=1,profile='velocity_slice')
    config=config.replace("schedule_mode='steps', step_interval=1, initial_frame=f, final_frame=t,",'')
    changed=config.replace('&insitu_run',"&insitu_run schedule_mode='time',time_interval=0.75,"
        "initial_frame=t,final_frame=t,product_ids='not-a-native-product.image',"
        "product_modes='steps',product_steps=3,slice_axis='unused',slice_index=-1,"
        "slice_definition='unused',slice_origin=1,2,3,streamline_seeds='unused',"
        "streamline_views_file='not-read.nml',wall_mean_render=t,mean_streamline_render=t,")
    items=[dynamic_image('streamlines','lines'),dynamic_image('slice','plane')]
    for item in items: item['schedule']['interval']=4
    common=dict(grid='32,32,32',insitu_config=config,checkpoint_interval=5,product_catalog_records=items)
    full,_=run_case(args,ROOT,'gpu',2,'full',12,**common)
    prefix,_=run_case(args,ROOT,'gpu',2,'prefix',5,**common)
    tail,_=run_case(args,ROOT,'gpu',2,'tail',12,**dict(common,insitu_config=changed,product_catalog_records=[]),
        restore=prefix/'outdat/new/checkpoints/step000000000005')
    final='outdat/new/checkpoints/step000000000012'
    for name in ('state.h5','statistics.h5'): compare_fields(full/final/name,tail/final/name)
    status=json.loads((tail/'outdat/new/products.status.json').read_text())
    assert {row['id'] for row in status['products']}=={'lines','plane'},status
    expected={path.parents[1].name:path for path in image_segments(full)}
    for path in image_segments(tail):
        files=sorted(path.glob('*.jpeg'));assert [int(file.name[10:22]) for file in files]==[8,12]
        for file in files:
            with Image.open(file) as a,Image.open(expected[path.parents[1].name]/file.name) as b:
                np.testing.assert_array_equal(a,b)
    assert not list((tail/'outdat/render').glob('*.jpeg'))
