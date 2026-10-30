"""Publish final native polydata; no clocks, field computations or flow snapshots."""
import errno
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from vtkmodules.vtkCommonCore import vtkDoubleArray,vtkFloatArray,vtkTypeInt64Array,vtkStringArray,vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter


def tagged_polydata(piece,context,step,time,namespace):
    points=int(namespace['metadata'](piece,'device_points'))
    cells=int(namespace['metadata'](piece,'device_cells'))
    arity=int(namespace['metadata'](piece,'device_arity'))
    width=int(namespace['metadata'](piece,'source_real_bytes'))
    poly=vtkPolyData()
    if points:
        poly.SetPoints(piece.GetPoints())
    else:
        empty=vtkPoints()
        empty.SetDataTypeToDouble() if width==8 else empty.SetDataTypeToFloat()
        poly.SetPoints(empty)
    if arity==3: poly.SetPolys(piece.GetCells())
    else: poly.SetLines(piece.GetCells())
    for name in context['point_fields']:
        data=piece.GetPointData().GetArray(name)
        if points==0:
            data=vtkDoubleArray() if width==8 else vtkFloatArray()
            data.SetName(name);data.SetNumberOfTuples(0)
        if data is None: raise RuntimeError('Native geometry is missing point field '+name)
        poly.GetPointData().AddArray(data)
    for name in context.get('cell_fields',[]):
        data=piece.GetCellData().GetArray(name)
        if cells==0:
            data=vtkTypeInt64Array();data.SetName(name);data.SetNumberOfTuples(0)
        if data is None: raise RuntimeError('Native geometry is missing cell field '+name)
        poly.GetCellData().AddArray(data)
    for name,value,array_type in (
        ('complete_step',step,vtkTypeInt64Array),('simulation_time',time,vtkDoubleArray),
        ('TimeValue',time,vtkDoubleArray),('definition_version',context['version'],vtkTypeInt64Array)):
        data=array_type();data.SetName(name);data.InsertNextValue(value);poly.GetFieldData().AddArray(data)
    references=vtkDoubleArray();references.SetName('reference_state')
    for value in context['reference_state']: references.InsertNextValue(value)
    poly.GetFieldData().AddArray(references)
    labels=dict(product_id=context['id'],source=context['source'],sampling_phase='completed_step',
        processing_backend=context['processing_backend'],
        unit_system=context['unit_system'],field_units=json.dumps(context['units'],sort_keys=True),
        reference_state_names='rhoinf,uinf,vinf,winf,pinf,tinf,ref_den,ref_vel,ref_tem,ref_len,ref_tim,ref_miu')
    if 'statistics' in context: labels['statistics']=json.dumps(context['statistics'],sort_keys=True)
    for name,value in labels.items():
        data=vtkStringArray();data.SetName(name);data.InsertNextValue(value);poly.GetFieldData().AddArray(data)
    return poly,points,cells,width


def manifest(poly,context,parts,width):
    root=ET.Element('VTKFile',type='PPolyData',version='1.0',byte_order='LittleEndian',header_type='UInt64')
    parallel=ET.SubElement(root,'PPolyData',GhostLevel='0')
    # Let VTK encode its own string/numeric metadata, without serializing any geometry here.
    metadata=vtkPolyData();metadata.GetFieldData().ShallowCopy(poly.GetFieldData())
    writer=vtkXMLPolyDataWriter();writer.SetInputData(metadata);writer.SetDataModeToAscii();writer.WriteToOutputStringOn()
    if not writer.Write() or writer.GetErrorCode(): raise OSError(errno.EIO,'VTK metadata serialization failed')
    parallel.append(ET.fromstring(writer.GetOutputString()).find('PolyData/FieldData'))
    for tag,names,kind in (
        ('PPointData',context['point_fields'],'Float64' if width==8 else 'Float32'),
        ('PCellData',context.get('cell_fields',[]),'Int64')):
        attributes=ET.SubElement(parallel,tag)
        for name in names: ET.SubElement(attributes,'PDataArray',type=kind,Name=name,NumberOfComponents='1')
    coordinates=ET.SubElement(parallel,'PPoints')
    ET.SubElement(coordinates,'PDataArray',type='Float64' if width==8 else 'Float32',NumberOfComponents='3')
    for part in parts: ET.SubElement(parallel,'Piece',Source=part)
    return ET.tostring(root,encoding='utf-8',xml_declaration=True)


def publish_geometry(context,piece,namespace,step,time):
    dispatch,controller=namespace['dispatch'],namespace['controller']
    poly,points,cells,width=dispatch.collective_call(controller,lambda:tagged_polydata(piece,context,step,time,namespace))
    rank=namespace['rank'];ranks=controller.GetNumberOfProcesses();directory=Path(context['directory'])
    base=f'frame.step{step:012d}'
    destination=directory/f'{base}.rank{rank:08d}.vtp'
    owned=set()

    def publish_partial(temporary,final):
        os.link(temporary,final);owned.add(final)
        temporary.unlink();owned.remove(temporary)

    def write_part():
        temporary=destination.with_name(destination.name+'.partial')
        with temporary.open('xb'): owned.add(temporary)
        writer=vtkXMLPolyDataWriter();writer.SetFileName(str(temporary));writer.SetInputData(poly)
        writer.SetHeaderTypeToUInt64()
        if not writer.Write() or writer.GetErrorCode(): raise OSError(errno.EIO,'VTK piece write/close failed')
        publish_partial(temporary,destination)

    def write_manifest():
        if rank!=0: return
        final=directory/f'{base}.pvtp';temporary=final.with_name(final.name+'.partial')
        payload=manifest(poly,context,[f'{base}.rank{i:08d}.vtp' for i in range(ranks)],width)
        with temporary.open('xb') as stream:
            owned.add(temporary);stream.write(payload)
        publish_partial(temporary,final)

    try:
        dispatch.collective_call(controller,write_part)
        dispatch.collective_call(controller,write_manifest)
    except dispatch.ProductFailure:
        def cleanup():
            for path in owned: path.unlink(missing_ok=True)
        dispatch.collective_call(controller,cleanup)
        raise
    return dict(geometry='published',manifest=base+'.pvtp',piece=destination.name,
        local_points=points,local_cells=cells,source_bytes=width,fields=context['point_fields'],
        cell_fields=context.get('cell_fields',[]),unit_system=context['unit_system'],units=context['units'],
        reference_state=context['reference_state'])
