#!/usr/bin/env pvpython
"""Known plane/cube display reference; no solver, state or flow-field arrays."""
import argparse
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from paraview import simple as pv
from vtkmodules.vtkFiltersSources import vtkPlaneSource, vtkCubeSource
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/insitu"))
from render_coordinates import update_coordinates,update_static_coordinates


def render(output, name, flags, values, volume=False,metadata_only=False,static=False,plane_axis=2,require_plane_layout=False):
    source = vtkCubeSource() if volume else vtkPlaneSource()
    if volume:
        source.SetBounds(10., 30., 0., 2., 0., 4.)
    else:
        if plane_axis==2:
            source.SetOrigin(10., 0., 0.); source.SetPoint1(30., 0., 0.); source.SetPoint2(10., 2., 0.)
        elif plane_axis==1:
            source.SetOrigin(10., 0., 0.); source.SetPoint1(30., 0., 0.); source.SetPoint2(10., 0., 2.)
        else:
            source.SetOrigin(0., 10., 0.); source.SetPoint1(0., 30., 0.); source.SetPoint2(0., 10., 2.)
    source.Update()
    before = [source.GetOutput().GetPoint(i) for i in range(source.GetOutput().GetNumberOfPoints())]
    mapper, actor = vtkPolyDataMapper(), vtkActor()
    mapper.SetInputConnection(source.GetOutputPort()); actor.SetMapper(mapper)
    actor.GetProperty().SetColor(.2, .45, .7); actor.GetProperty().SetLighting(False)
    view = pv.CreateView("RenderView")
    view.ViewSize = [1000, 700]; view.UseColorPaletteForBackground = 0; view.Background = [1., 1., 1.]
    view.OrientationAxesVisibility = 0; view.CameraParallelProjection = 1
    view.CameraPosition = [40., -22., 28.] if volume else [20., 1., 30.]
    view.CameraFocalPoint = [20., 1., 2.] if volume else [20., 1., 0.]
    view.CameraViewUp = [0., 0., 1.] if volume else [0., 1., 0.]
    view.CameraParallelScale = 13.
    if not volume and plane_axis!=2:
        view.CameraPosition = [20., -30., 1.] if plane_axis==1 else [30., 20., 1.]
        view.CameraFocalPoint = [20., 0., 1.] if plane_axis==1 else [0., 20., 1.]
        view.CameraViewUp = [0., 0., 1.]
    view.UpdateVTKObjects()
    client = view.GetClientSideObject(); renderer = client.GetRenderer(); renderer.AddActor(actor)
    item = dict(view=view, actor=actor, coordinate_options=(flags, values, "m"))
    bounds = [10., 30., 0., 2., 0., 4.] if volume else [10., 30., 0., 2., 0., 0.]
    if not volume and plane_axis!=2:
        bounds=[10.,30.,0.,0.,0.,2.] if plane_axis==1 else [0.,0.,10.,30.,0.,2.]
    camera_before=(list(view.CameraPosition),list(view.CameraFocalPoint),list(view.CameraViewUp),view.CameraParallelScale)
    if static:
        def reject_clipping(*args):
            raise AssertionError('static annotation must preserve caller clipping bounds')
        annotation_client=SimpleNamespace(GetRenderer=client.GetRenderer,SetMaxClipBounds=reject_clipping)
        item['view']=SimpleNamespace(ViewSize=view.ViewSize,GetClientSideObject=lambda:annotation_client,
            CameraPosition=list(view.CameraPosition),CameraFocalPoint=list(view.CameraFocalPoint),
            CameraViewUp=list(view.CameraViewUp),CameraParallelScale=view.CameraParallelScale)
        display_bounds=[10.,30.,0.,2.,0.,4.]
        plane=None if volume else dict(origin=[0.,0.,0.],normal=[0.,0.,1.])
        model=update_static_coordinates(item,None,display_bounds,planar=not volume,plane=plane)
        assert camera_before==(list(view.CameraPosition),list(view.CameraFocalPoint),list(view.CameraViewUp),view.CameraParallelScale)
        count=renderer.GetViewProps().GetNumberOfItems()
        changed=[11.,29.,.2,1.8,0.,4.]
        model=update_static_coordinates(item,None,changed,planar=not volume,plane=plane)
        assert renderer.GetViewProps().GetNumberOfItems()==count
        assert model['physical_bounds'][:4]==changed[:4]
        update_static_coordinates(item,None,changed,planar=not volume,plane=plane,empty=True)
        assert not item['coordinate_axes'].GetVisibility()
        model=update_static_coordinates(item,None,changed,planar=not volume,plane=plane)
        assert item['coordinate_axes'].GetVisibility()
        annotation_view=item['view']
        assert camera_before==(annotation_view.CameraPosition,annotation_view.CameraFocalPoint,
                               annotation_view.CameraViewUp,annotation_view.CameraParallelScale)
    else:
        model = update_coordinates(item, None, "{}", bounds, fixed_axis=None if volume else plane_axis)
    after = [source.GetOutput().GetPoint(i) for i in range(source.GetOutput().GetNumberOfPoints())]
    assert before == after
    axes = item["coordinate_axes"]
    assert all(getattr(axes, f"GetDraw{axis}Gridlines")() == 0 for axis in "XYZ")
    assert list(axes.GetXAxisRange()) == model["axis_ranges"][0]
    assert list(axes.GetYAxisRange()) == model["axis_ranges"][1]
    assert list(axes.GetZAxisRange()) == model["axis_ranges"][2]
    assert list(axes.GetBounds()) == model["display_bounds"]
    if require_plane_layout:
        assert bool(axes.GetUse2DMode())==(not volume and plane_axis==2), 'VTK 2D mode assumes world XY; XZ/YZ need the 3D axis followers'
    if not metadata_only:
        pv.Render(view)
        pv.SaveScreenshot(str(output / (name + ".jpeg")), view, ImageResolution=[1000, 700], Quality=95)
    model["input_points_unchanged"] = True
    model["vtk_xy_only_2d_mode"] = bool(axes.GetUse2DMode())
    model["render_window"] = client.GetRenderWindow().GetClassName()
    if not metadata_only:
        (output / (name + ".json")).write_text(json.dumps(model, indent=2) + "\n")
    for prop in item["coordinate_actors"]:
        prop.ReleaseGraphicsResources(client.GetRenderWindow()); renderer.RemoveViewProp(prop)
    renderer.RemoveActor(actor); pv.Delete(view)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--metadata-only",action='store_true',help='inspect real VTK props without Render or file output')
    parser.add_argument("--collective-only",action='store_true',help='test only the 64-bit geometry count with pvbatch --symmetric')
    parser.add_argument('--include-wall-planes',action='store_true')
    parser.add_argument('--require-plane-layout',action='store_true')
    args = parser.parse_args()
    if args.collective_only:
        from vtkmodules.vtkCommonCore import vtkTypeInt64Array,vtkDoubleArray
        from vtkmodules.vtkParallelCore import vtkMultiProcessController,vtkCommunicator
        controller=vtkMultiProcessController.GetGlobalController()
        rank,ranks=controller.GetLocalProcessId(),controller.GetNumberOfProcesses()
        local,total=vtkTypeInt64Array(),vtkTypeInt64Array()
        local.InsertNextValue(0 if rank==0 and ranks>1 else (1<<35)+rank)
        total.SetNumberOfTuples(1)
        controller.AllReduce(local,total,vtkCommunicator.SUM_OP)
        assert total.GetValue(0)==sum((1<<35)+r for r in range(ranks) if r>0 or ranks==1)
        local.SetValue(0,0)
        controller.AllReduce(local,total,vtkCommunicator.SUM_OP)
        assert total.GetValue(0)==0
        pipeline=Path(__file__).resolve().parents[2]/'scripts/insitu/device_render_pipeline.py'
        functions=[node for node in ast.parse(pipeline.read_text()).body if
                   isinstance(node,ast.FunctionDef) and node.name in ('metadata','global_device_bounds')]
        namespace=dict(vtkDoubleArray=vtkDoubleArray,vtkCommunicator=vtkCommunicator,controller=controller)
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(pipeline),'exec'),namespace)
        piece=vtkPolyData()
        active=rank>0 or ranks==1
        values={'device_points':2 if active else 0}
        values.update({'bound'+str(d):v for d,v in enumerate([2.*rank,2.*rank+2.,1.,1.,3.,3.])})
        for key,value in values.items():
            array=vtkDoubleArray();array.SetName(key);array.InsertNextValue(value);piece.GetFieldData().AddArray(array)
        bounds=namespace['global_device_bounds'](piece,allow_planar=True,allow_empty=True,minimum_dimensions=0)
        assert bounds==[0. if ranks==1 else 2.,2.*ranks,1.,1.,3.,3.]
        try:
            namespace['global_device_bounds'](piece,allow_planar=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError('the existing plane admission must still require two-dimensional extent')
        piece.GetFieldData().GetArray('device_points').SetValue(0,0.)
        assert namespace['global_device_bounds'](piece,allow_planar=True,allow_empty=True,minimum_dimensions=0) is None
        print('PASS VTK 64-bit geometry count, empty rank and actual global bounds helper rank='+str(rank))
        return
    if not args.metadata_only:
        if args.out is None:
            parser.error('--out is required when writing images')
        args.out.mkdir(parents=True,exist_ok=False)
    common=dict(metadata_only=args.metadata_only,require_plane_layout=args.require_plane_layout)
    render(args.out, "plane-native", [1, 0], [0., 0., 0., 1., 1., 1., 1.],**common)
    render(args.out, "plane-normalized", [1, 1], [10., 0., 0., 2., 1., 1., 1.],**common)
    render(args.out, "plane-stretched", [1, 0], [0., 0., 0., 1., 1., 4., 1.],**common)
    render(args.out, "cube-native", [1, 0], [0., 0., 0., 1., 1., 1., 1.], volume=True,**common)
    render(args.out, "static-plane", [1, 0], [0., 0., 0., 1., 1., 1., 1.],static=True,**common)
    render(args.out, "static-volume", [1, 0], [0., 0., 0., 1., 1., 1., 1.],static=True,volume=True,**common)
    if args.include_wall_planes:
        for axis,name in [(1,'xz'),(0,'yz')]:
            render(args.out,name+'-native',[1,0],[0.,0.,0.,1.,1.,1.,1.],plane_axis=axis,**common)
            render(args.out,name+'-stretched',[1,0],[0.,0.,0.,1.,1.,1.,4.],plane_axis=axis,**common)
    print("PASS VTK coordinate ranges, fixed-plane axes, input geometry, display transform and static view reuse; metadata_only="+str(args.metadata_only))


if __name__=='__main__':
    main()
