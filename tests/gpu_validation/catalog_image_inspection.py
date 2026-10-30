"""Inspect the native product files, never infer extra products from a profile."""
import json
from pathlib import Path

from PIL import Image


def inspect_images(case,ranks,steps):
    case=Path(case)
    definitions=[json.loads(line) for line in (case/'input.products.jsonl').read_text().splitlines()[1:] if line.strip()]
    images=[row for row in definitions if row['output']['channel']=='image']
    # Some standalone tools choose another output root in their input.output.
    candidates=list((case/'outdat').rglob('products.status.json'))
    if len(candidates)!=1: raise ValueError('Expected one native product status file')
    root=candidates[0].parent
    statuses={row['id']:row for row in json.loads(candidates[0].read_text())['products']}
    frames=[];warnings=[]
    for row in images:
        name=row['id'];directory=root/'render_products'/name
        if statuses.get(name,{}).get('state')!='active': warnings.append(f'Inactive product: {name}')
        paths=sorted(directory.glob('version*/segment*/frame.step*.jpeg'))
        expected=set(range(0,steps+1,row['schedule']['interval'])) if (
            row['source']=='instantaneous' and row['schedule']['mode']=='steps') else None
        actual=set()
        for path in paths:
            step=int(path.stem.removeprefix('frame.step'));actual.add(step)
            for rank in range(ranks):
                receipt=path.with_name(f'{path.stem}.rank{rank:08d}.json')
                if not receipt.is_file():
                    warnings.append(f'Missing product receipt: {receipt.relative_to(case)}');continue
                record=json.loads(receipt.read_text())
                if record['id']!=name or record['definition']!=row or record['phase']!='completed_step':
                    raise ValueError('Native product definition or phase differs')
                if record['processing_backend']!='device' or record['field_d2h_bytes'] or record['geometry_host_bytes']:
                    raise ValueError('Product pipeline used host geometry/fields or non-device processing')
                if record['product']['image']!='published': warnings.append(f'Image not published: {name}/{step}/{rank}')
                if record.get('geometry_empty'): warnings.append(f'{name}/{step}/{rank}: empty geometry (valid diagnostic outcome)')
                frames.append(dict(id=name,step=step,rank=rank))
            if 'eps' in row['output']['formats'] and not path.with_suffix('.eps').is_file():
                warnings.append(f'Missing EPS image: {name}/{step}')
        if not actual or (expected is not None and actual!=expected):
            warnings.append(f'Missing JPEG image or unexpected sequence: {name}; actual={sorted(actual)} expected={sorted(expected) if expected is not None else "at least one covered frame"}')
        for path in dict.fromkeys(paths[:1]+paths[-1:]):
            try:
                with Image.open(path) as image:
                    image.load()
                    if list(image.size)!=row['output']['image_size']: raise ValueError('image size differs')
            except (OSError,ValueError) as error:
                warnings.append(f'Image decode failed: {name}: {error}')
    if list(root.rglob('*.vtp')): raise ValueError('Unexpected host geometry dump in the image gate')
    return dict(products=[row['id'] for row in images],frames=frames,warnings=warnings,
        publication_complete=all('empty geometry' in warning for warning in warnings))
