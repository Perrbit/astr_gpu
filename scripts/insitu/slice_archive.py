"""Read native-precision fixed-index ASTR slices; never reconstruct absent 3D data."""
import argparse
import copy
import csv
from decimal import Decimal
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def read_slice(path, selected_fields=None):
    path = Path(path)
    with h5py.File(path) as frame:
        definition = json.loads(text(frame.attrs['product_definition']))
        names = definition['fields'] if selected_fields is None else selected_fields
        fields = {name: frame[name][...] for name in names}
        units = {name: text(frame[name].attrs['units']) for name in names}
        geometry_path = path.parent / text(frame.attrs['geometry_file'])
        result = dict(step=int(frame['step'][0]), time=float(frame['clock'][0]),
            ranges=frame['index_ranges'][...].reshape(3, 2), fields=fields, units=units,
            definition=definition, source_measure=text(frame.attrs['source_measure']))
        if 'reference_state' in frame:
            result['reference_state'] = dict(zip(text(frame.attrs['reference_state_layout']).split(),
                                                 frame['reference_state'][...].tolist(), strict=True))
    with h5py.File(geometry_path) as geometry:
        result['coordinates'] = geometry['coordinates'][...]
    return result


def describe(path):
    with h5py.File(path) as frame:
        definition = json.loads(text(frame.attrs['product_definition']))
        return dict(path=str(path), step=int(frame['step'][0]), time=float(frame['clock'][0]),
            fields={name: dict(shape=list(frame[name].shape), dtype=str(frame[name].dtype),
                               units=text(frame[name].attrs['units'])) for name in definition['fields']},
            source_measure=text(frame.attrs['source_measure']))


def write_profile(path, field, dimension, index, output):
    if dimension not in (0, 1):
        raise ValueError('profile dimension must be 0 or 1 of the saved two-dimensional array')
    selection = [slice(None), slice(None)]
    selection[dimension] = index
    with h5py.File(path) as frame:
        if not 0 <= index < frame[field].shape[dimension]:
            raise ValueError('profile index lies outside the saved rectangle')
        values = frame[field][tuple(selection)]
        geometry_path = Path(path).parent/text(frame.attrs['geometry_file'])
    with h5py.File(geometry_path) as geometry:
        coordinates = geometry['coordinates'][tuple(selection)]
    with Path(output).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['x', 'y', 'z', field])
        writer.writerows(np.column_stack([coordinates, values]).tolist())


def plot_slice(path, field, output, axes=None, limits=None, eps=False, window=None):
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'ieee', 'std-colors'])
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.alpha'] = 0.0
    plt.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize': 14,
                         'ytick.labelsize': 14, 'legend.fontsize': 14, 'text.usetex': False})
    data = read_slice(path, [field])
    fixed = 'ijk'.index(data['definition']['sampling']['axis'])
    dimensions = [i for i in range(3) if i != fixed] if axes is None else ['xyz'.index(a) for a in axes]
    coordinates = data['coordinates']
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    color = {} if limits is None else dict(vmin=limits[0], vmax=limits[1])
    image = ax.pcolormesh(coordinates[..., dimensions[0]], coordinates[..., dimensions[1]],
        data['fields'][field], cmap='RdBu_r', shading='nearest', **color)
    ax.set_xlabel('xyz'[dimensions[0]])
    ax.set_ylabel('xyz'[dimensions[1]])
    if window is not None:
        ax.set_xlim(window[:2]);ax.set_ylim(window[2:])
    fig.colorbar(image, ax=ax, label=field + ' [' + data['units'][field] + ']')
    output = Path(output)
    fig.savefig(output.with_suffix('.jpeg'), dpi=300)
    if eps:
        fig.savefig(output.with_suffix('.eps'))
    plt.close(fig)


def join_segments(segments, output):
    """Build explicit external references; this does not make a self-contained copy."""
    output = Path(output).absolute()
    root = ET.Element('Xdmf', Version='3.0')
    domain = ET.SubElement(root, 'Domain')
    collection = ET.SubElement(domain, 'Grid', Name='series', GridType='Collection', CollectionType='Temporal')
    definition_key = None
    previous = None
    for segment in segments:
        segment = Path(segment).absolute()
        if segment.is_file():
            segment = segment.parent
        definition = json.loads((segment/'definition.json').read_text())
        key = (definition['source'], definition.get('statistics_source', ''),
               sorted(definition['fields']), definition['sampling'])
        if definition_key is not None and key != definition_key:
            raise ValueError('segments have different numerical slice definitions')
        definition_key = key
        source = ET.parse(segment/'series.xdmf')
        for old in source.findall('./Domain/Grid/Grid'):
            grid = copy.deepcopy(old)
            time = Decimal(grid.find('Time').attrib['Value'])
            if previous is not None and time <= previous:
                raise ValueError('explicit segments must have increasing, nonoverlapping output times')
            previous = time
            for item in grid.findall('.//DataItem[@Format="HDF"]'):
                file, separator, dataset = item.text.strip().partition(':/')
                if not separator:
                    raise ValueError('malformed HDF reference in source XDMF')
                item.text = os.path.relpath(segment/file, output.parent) + ':/' + dataset
            collection.append(grid)
    ET.indent(root)
    with output.open('xb') as stream:
        ET.ElementTree(root).write(stream, encoding='utf-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list')
    listing.add_argument('path', type=Path)
    joining = commands.add_parser('join')
    joining.add_argument('segments', type=Path, nargs='+')
    joining.add_argument('--output', type=Path, required=True)
    for command in ('plot', 'profile'):
        sub = commands.add_parser(command)
        sub.add_argument('path', type=Path)
        sub.add_argument('--field', required=True)
        sub.add_argument('--output', type=Path, required=True)
        if command == 'profile':
            sub.add_argument('--dimension', type=int, choices=(0, 1), required=True)
            sub.add_argument('--index', type=int, required=True)
        else:
            sub.add_argument('--axes', nargs=2, choices=('x', 'y', 'z'))
            sub.add_argument('--limits', nargs=2, type=float)
            sub.add_argument('--window', nargs=4, type=float, metavar=('XMIN','XMAX','YMIN','YMAX'))
            sub.add_argument('--eps', action='store_true')
    args = parser.parse_args()
    if args.command == 'join':
        join_segments(args.segments, args.output)
    elif args.command == 'list':
        paths = sorted(args.path.glob('step*.h5')) if args.path.is_dir() else [args.path]
        print(json.dumps([describe(path) for path in paths], ensure_ascii=False, indent=2))
    elif args.command == 'profile':
        write_profile(args.path, args.field, args.dimension, args.index, args.output)
    else:
        if args.axes is not None and len(set(args.axes)) != 2:
            parser.error('--axes requires two different physical coordinate axes')
        plot_slice(args.path, args.field, args.output, args.axes, args.limits, args.eps, args.window)


if __name__ == '__main__':
    main()
