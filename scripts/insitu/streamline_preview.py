"""Compact seed diagnostics and single-view export; no flow or trajectory I/O."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def retain_display_resources(item):
    resources=item.setdefault('candidate_resources',{})
    for name in ('legend','lut'):
        proxy=item[name]
        resources[proxy.SMProxy.GetGlobalIDAsString()]=proxy
    return resources


def seed_projection(candidate, image_size=(800, 600)):
    values, sites = candidate['values'], candidate['seed_sites']
    eye, focal, up = values[:3], values[3:6], values[6:9]

    def normalized(vector):
        length = math.hypot(*vector)
        if not math.isfinite(length) or length == 0.:
            raise ValueError('Degenerate projection basis')
        return [x/length for x in vector]

    def cross(a, b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

    direction = normalized([focal[d]-eye[d] for d in range(3)])
    right = normalized(cross(direction, normalized(up)))
    vertical = cross(right, direction)
    bounds, scale = values[10:16], values[9]
    pixels = []
    for site in sites:
        if all(bounds[2*d] <= site[d] <= bounds[2*d+1] for d in range(3)):
            offset = [site[d]-focal[d] for d in range(3)]
            pixels.append([image_size[0]/2 + image_size[1]/(2*scale)*sum(offset[d]*right[d] for d in range(3)),
                           image_size[1]/2 + image_size[1]/(2*scale)*sum(offset[d]*vertical[d] for d in range(3))])
    nearest = [min(math.dist(p, q) for j, q in enumerate(pixels) if i != j)
               for i, p in enumerate(pixels)] if len(pixels) > 1 else []
    return dict(seed_sites_in_display_band=len(pixels), seed_sites_total=len(sites),
        seed_sites_in_viewport=sum(0 <= p[0] <= image_size[0] and 0 <= p[1] <= image_size[1] for p in pixels),
        projected_nearest_spacing_pixels=None if not nearest else [min(nearest), max(nearest)],
        projected_coincident_pairs=sum(math.dist(p, q) <= 1e-10 for i, p in enumerate(pixels) for q in pixels[i+1:]),
        seed_layers_y=[dict(y=y, count=sum(site[1] == y for site in sites)) for y in sorted({s[1] for s in sites})],
        interpretation='Seed projection only; not a count of visible or unoccluded trajectories')


def selected_namelist(record):
    view = record['candidate']
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,63}', view['id']):
        raise ValueError('Invalid selected candidate ID')
    for field, allowed in [('velocity', ('instantaneous', 'reynolds', 'favre')),
                           ('seeds', ('line16', 'tgv-stratified', 'bl-layered64', 'bl-layered256')),
                           ('color', ('u', 'v', 'w', 'speed'))]:
        if view[field] not in allowed:
            raise ValueError('Unknown selected '+field)
    values = view['values']
    if len(values) != 19 or not all(math.isfinite(x) for x in values):
        raise ValueError('Invalid selected numeric values')
    text = '&streamline_views\n'
    for field, name in [('id', 'candidate_ids'), ('velocity', 'velocity'), ('seeds', 'seeds'), ('color', 'color_field')]:
        text += f" {name}='{view[field]}',\n"
    for name, lo, hi in [('camera_position', 0, 3), ('focal_point', 3, 6), ('view_up', 6, 9),
                         ('display_bounds', 10, 16), ('color_range', 16, 18)]:
        text += f' {name}(:,1)='+','.join(format(x, '.17e') for x in values[lo:hi])+',\n'
    text += f' parallel_scale(1)={values[9]:.17e}, line_width(1)={values[18]:.17e}\n/\n'
    return text


def export_selection(receipt, destination):
    receipt, destination = Path(receipt), Path(destination)
    record = json.loads(receipt.read_text())
    if record['rendering']['image'] != 'published' or record['geometry_host_bytes'] != 0 or not record.get('source'):
        raise ValueError('Selection requires a published resident preview and its checkpoint identity')
    if (record['step'], record['time']) != (record['source']['step'], record['source']['time']):
        raise ValueError('Preview/source phase mismatch')
    if record['candidate']['velocity'] != 'instantaneous' and record.get('statistics', {}).get('statistics_duration', 0.) <= 0.:
        raise ValueError('Selection lacks mean coverage')
    sidecar = destination.with_suffix(destination.suffix+'.json')
    if destination.exists() or sidecar.exists():
        raise FileExistsError('Selected configuration must not overwrite an existing artifact')
    text = selected_namelist(record)
    provenance = {key: record[key] for key in ('source', 'step', 'time', 'candidate')}
    provenance.update(statistics=record.get('statistics'), receipt=str(receipt.resolve()),
        configuration_sha256=hashlib.sha256(text.encode()).hexdigest(),
        note='Configuration only; does not restore preview clocks or production history')
    with destination.open('x') as stream:
        stream.write(text)
    with sidecar.open('x') as stream:
        json.dump(provenance, stream, indent=2)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    export_selection(args.receipt, args.destination)
