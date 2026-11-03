"""Display-only coordinate scales and VTK axes; never transform pipeline data."""
import ctypes
import math


def native_options(native, definition):
    function = native.astr_catalog_display_coordinates
    function.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int),
                         ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_size_t]
    function.restype = ctypes.c_int
    text = definition.encode("utf-8")
    flags, values, error = (ctypes.c_int * 2)(), (ctypes.c_double * 7)(), ctypes.create_string_buffer(256)
    if function(text, len(text), flags, values, error, len(error)) != 1:
        raise ValueError(error.value.decode("utf-8"))
    return list(flags), list(values)


def native_unit(native):
    function = native.astr_insitu_runtime_geometry_source
    function.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                         ctypes.c_char_p, ctypes.c_char_p]
    function.restype = None
    ids, references = (ctypes.c_int * 1)(), (ctypes.c_double * 12)()
    system, units = ctypes.create_string_buffer(16), ctypes.create_string_buffer(1)
    function(ids, 0, references, system, units)
    return {"si": "m", "dimensionless": "1"}[system.raw.decode("ascii").strip(" \0")]


def fixed_physical_axis(sampling, normal, bounds):
    if sampling["mode"] == "grid_index":
        axis = "ijk".index(sampling["axis"])
        return axis if bounds[2*axis] == bounds[2*axis+1] else None
    directions = [d for d, value in enumerate(normal) if value != 0.]
    return directions[0] if len(directions) == 1 else None


def static_coordinate_frame(bounds, planar=False, plane=None):
    frame=list(bounds)
    fixed=None
    if planar:
        if plane is not None:
            fixed=fixed_physical_axis({'mode':'physical_plane'},plane['normal'],frame)
            if fixed is not None:
                frame[2*fixed:2*fixed+2]=[plane['origin'][fixed]]*2
        else:
            collapsed=[d for d in range(3) if frame[2*d]==frame[2*d+1]]
            if len(collapsed)==1:
                fixed=collapsed[0]
    return frame,fixed


def update_static_coordinates(item,native,bounds,planar=False,plane=None,empty=False):
    if 'coordinate_options' not in item:
        item['coordinate_options']=([1,0],[0.,0.,0.,1.,1.,1.,1.],native_unit(native))
    frame,fixed=static_coordinate_frame(bounds,planar,plane)
    model=update_coordinates(item,native,'{}',frame,fixed_axis=fixed,empty=empty,manage_clipping=False)
    model['bounds_role']='caller-supplied physical frame; oblique planes retain global Cartesian axes'
    return model


def coordinate_model(flags, values, bounds, unit, fixed_axis=None):
    origin, length, scale = list(values[:3]), values[3], list(values[4:7])
    display = [origin[d] + scale[d] * (bounds[2*d+s] - origin[d]) for d in range(3) for s in (0, 1)]
    ranges = [[(bounds[2*d+s] - origin[d]) / length if flags[1] else bounds[2*d+s] for s in (0, 1)]
              for d in range(3)]
    if not all(math.isfinite(value) for value in display + sum(ranges, [])):
        raise ValueError("coordinate display transform or tick range is not finite")
    titles = [f"({axis}-{origin[d]:g})/{length:g} [1]" if flags[1] else f"{axis} [{unit}]"
              for d, axis in enumerate("xyz")]
    note = "; ".join(f"{axis} display x{scale[d]:g}" for d, axis in enumerate("xyz") if scale[d] != 1.)
    return dict(visible=bool(flags[0]), axis_visible=[d != fixed_axis for d in range(3)],
                native_coordinate_unit=unit, physical_bounds=list(bounds), display_bounds=display,
                axis_ranges=ranges, axis_titles=titles, origin=origin,
                normalized=bool(flags[1]), reference_length=length, display_scale=scale,
                display_note=note, gridlines=False,
                geometry_transform="actor only; camera/focal points use the affine map; parallel scale stays explicit")


def transform_camera(position, focal, up, parallel_scale, model):
    origin, scale = model["origin"], model["display_scale"]
    def point(values):
        return [origin[d] + scale[d] * (values[d] - origin[d]) for d in range(3)]
    scaled_up = [scale[d] * up[d] for d in range(3)]
    up_size = math.hypot(*scaled_up)
    if up_size == 0. or not math.isfinite(up_size):
        raise ValueError("display-scaled camera up vector is not representable")
    result = dict(position=point(position), focal_point=point(focal),
                  view_up=[value / up_size for value in scaled_up],
                  parallel_scale=parallel_scale)
    if not all(math.isfinite(v) for key in ("position", "focal_point", "view_up") for v in result[key]) or \
            not math.isfinite(result["parallel_scale"]):
        raise ValueError("display-scaled camera is not finite")
    return result


def update_coordinates(item, native, definition, bounds, fixed_axis=None, empty=False,manage_clipping=True):
    from vtkmodules.vtkCommonCore import vtkStringArray
    from vtkmodules.vtkRenderingAnnotation import vtkCubeAxesActor
    from vtkmodules.vtkRenderingCore import vtkTextActor
    if "coordinate_options" not in item:
        item["coordinate_options"] = (*native_options(native, definition), native_unit(native))
    flags, values, unit = item["coordinate_options"]
    model = coordinate_model(flags, values, bounds, unit, fixed_axis)
    model["visible_requested"] = model["visible"]
    model["visible"] = model["visible"] and not empty
    view = item["view"]
    renderer = view.GetClientSideObject().GetRenderer()
    if "coordinate_axes" not in item:
        axes, note = vtkCubeAxesActor(), vtkTextActor()
        axes.SetCamera(renderer.GetActiveCamera())
        axes.SetFlyModeToOuterEdges()
        axes.SetTickLocationToOutside()
        axes.SetEnableDistanceLOD(False)
        axes.SetEnableViewAngleLOD(False)
        axes.SetScreenSize(14.)
        axes.SetTitleOffset([35., 35.])
        for d, axis in enumerate("XYZ"):
            getattr(axes, f"Draw{axis}GridlinesOff")()
            getattr(axes, f"Draw{axis}InnerGridlinesOff")()
            getattr(axes, f"Draw{axis}GridpolysOff")()
            getattr(axes, f"Get{axis}AxesLinesProperty")().SetColor(0., 0., 0.)
            axes.GetTitleTextProperty(d).SetColor(0., 0., 0.)
            axes.GetLabelTextProperty(d).SetColor(0., 0., 0.)
            axes.GetTitleTextProperty(d).SetFontSize(16)
            axes.GetLabelTextProperty(d).SetFontSize(14)
            getattr(axes, f"Set{axis}LabelFormat")("%.4g")
        note.GetTextProperty().SetColor(0., 0., 0.)
        note.GetTextProperty().SetFontSize(14)
        note.SetDisplayPosition(18, 12)
        renderer.AddViewProp(axes); renderer.AddViewProp(note)
        item["coordinate_axes"], item["coordinate_note"] = axes, note
        item["coordinate_actors"] = [axes, note]
        item["coordinate_tick_labels"] = [None, None, None]
        item["actor"].SetOrigin(model["origin"])
        item["actor"].SetScale(model["display_scale"])
        if any(value != 1. for value in model["display_scale"]):
            camera = transform_camera(view.CameraPosition, view.CameraFocalPoint, view.CameraViewUp,
                                      float(view.CameraParallelScale), model)
            view.CameraPosition = camera["position"]; view.CameraFocalPoint = camera["focal_point"]
            view.CameraViewUp = camera["view_up"]; view.CameraParallelScale = camera["parallel_scale"]
            view.UpdateVTKObjects()
    axes = item["coordinate_axes"]
    axes.SetBounds(model["display_bounds"])
    # VTK's 2D label/title placement only implements world X/Y axes, not X/Z or Y/Z.
    axes.SetUse2DMode(fixed_axis == 2)
    axes.SetVisibility(model["visible"])
    size = list(view.ViewSize)
    matrix = renderer.GetActiveCamera().GetCompositeProjectionTransformMatrix(size[0]/size[1], 0., 1.)
    center = [.5*(model["display_bounds"][2*d]+model["display_bounds"][2*d+1]) for d in range(3)]
    range_notes = []
    model["axis_labels_in_note"] = []
    for d, axis in enumerate("XYZ"):
        getattr(axes, f"Set{axis}AxisRange")(*model["axis_ranges"][d])
        getattr(axes, f"Set{axis}Title")(model["axis_titles"][d])
        getattr(axes, f"Set{axis}AxisVisibility")(model["axis_visible"][d])
        endpoints = []
        for side in (0, 1):
            point = list(center); point[d] = model["display_bounds"][2*d+side]
            endpoints.append(matrix.MultiplyPoint([*point, 1.]))
        pixels = 0.
        if all(point[3] != 0. for point in endpoints):
            screen = [[point[c]/point[3]*size[c]*.5 for c in (0, 1)] for point in endpoints]
            pixels = math.hypot(*(screen[1][c]-screen[0][c] for c in (0, 1)))
        collapsed = pixels < 40. and d != fixed_axis
        getattr(axes, f"Set{axis}AxisLabelVisibility")(not collapsed)
        if collapsed:
            getattr(axes, f"Set{axis}Title")("")
            model["axis_labels_in_note"].append(d)
            lo, hi = model["axis_ranges"][d]
            range_notes.append(f'{model["axis_titles"][d]} range [{lo:.5g}, {hi:.5g}]')
        labels = tuple(f"{value:.4g}" for value in model["axis_ranges"][d]) if pixels < 160. and d != fixed_axis else None
        if labels != item["coordinate_tick_labels"][d]:
            array = None
            if labels is not None:
                array = vtkStringArray()
                for label in labels:
                    array.InsertNextValue(label)
            axes.SetAxisLabels(d, array)
            item["coordinate_tick_labels"][d] = labels
    notes = ([model["display_note"]] if model["display_note"] else []) + range_notes
    model["coordinate_note"] = "; ".join(notes)
    item["coordinate_note"].SetInput(model["coordinate_note"])
    item["coordinate_note"].SetVisibility(bool(notes) and not empty)
    if manage_clipping:
        clipping = list(model["display_bounds"])
        padding = .01 * max(clipping[2*d+1] - clipping[2*d] for d in range(3))
        for d in range(3):
            if clipping[2*d] == clipping[2*d+1]:
                clipping[2*d] -= padding; clipping[2*d+1] += padding
        view.GetClientSideObject().SetMaxClipBounds(clipping)
    item["coordinate_annotation"] = model
    return model
