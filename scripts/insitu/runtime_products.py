"""Consume one immutable native product descriptor; clocks remain in Fortran."""
import ctypes
import json
from pathlib import Path


class RuntimeProducts:
    def __init__(self):
        self.native = ctypes.CDLL(None)
        self.native.astr_insitu_runtime_context.argtypes = []
        self.native.astr_insitu_runtime_context.restype = ctypes.c_char_p
        self.native.astr_insitu_runtime_failure.argtypes = [ctypes.c_char_p]
        self.native.astr_insitu_runtime_failure.restype = None
        self.native.astr_insitu_release_runtime_actor.argtypes = [ctypes.c_char_p]
        self.native.astr_insitu_release_runtime_actor.restype = None
        self.versions = {}
        self.current = None

    def retire(self, name, namespace):
        if name in namespace['render_objects']:
            namespace['release_render_product'](name)
        self.native.astr_insitu_release_runtime_actor(name.encode('utf-8'))
        self.versions.pop(name, None)

    def setup_view(self, name, item, piece, namespace):
        context = self.current
        pv = namespace['pv']
        settings, words, labels = context['reals'], context['integers'], context['labels']
        view = pv.CreateView('RenderView')
        item['view'] = view
        view.ViewSize = words[:2]
        view.UseColorPaletteForBackground = 0
        view.Background = [1., 1., 1.]
        view.OrientationAxesVisibility = 0
        view.RemoteRenderThreshold = 0.
        view.CameraPosition = settings[12:15]
        view.CameraFocalPoint = settings[15:18]
        view.CameraViewUp = settings[18:21]
        view.CameraParallelProjection = int(labels[5] == 'parallel')
        if labels[5] == 'parallel':
            view.CameraParallelScale = settings[21]
        view.UpdateVTKObjects()
        item['actor'].GetProperty().SetLighting(bool(words[5]))
        if settings[24] > 0.:
            item['actor'].GetProperty().SetLineWidth(settings[24])
        index_surface = labels[3] == 'grid_index'
        glyph=labels[4]=='wall_shear_vector'
        wall=labels[4] in ('wall_shear_vector','wall_shear_lines')
        bounds = (namespace['global_device_bounds'](piece, allow_planar=True,
                    count_name='wall_source_nodes',bounds_prefix='wall_source_bound',minimum_dimensions=0) if wall else
                  namespace['global_device_bounds'](piece, allow_planar=True) if index_surface
                  else list(settings[6:12]))
        padding = .01 * max(bounds[2*d+1] - bounds[2*d] for d in range(3))
        if glyph:padding=max(padding,json.loads(context['definition'])['extraction']['length'])
        clip = list(bounds)
        for d in range(3):
            if clip[2*d] == clip[2*d+1]:
                clip[2*d] -= padding
                clip[2*d+1] += padding
        client = view.GetClientSideObject()
        client.GetRenderer().AddActor(item['actor'])
        client.SetMaxClipBounds(clip)
        client.SetLockBounds(True)
        lut = pv.GetColorTransferFunction('runtime_' + name + '_v' + str(context['version']))
        lut.AutomaticRescaleRangeMode = 'Never'
        if labels[6] == 'grayscale':
            lut.RGBPoints = [0., 0., 0., 0., 1., 1., 1., 1.]
            lut.ColorSpace = 'RGB'
        else:
            lut.ApplyPreset('Cool to Warm (Extended)', False)
        lut.RescaleTransferFunction(*settings[22:24])
        if namespace['pipeline']=='compatible':
            item['mapper'].SetLookupTable(lut.GetClientSideObject())
        legend = pv.GetScalarBar(lut, view)
        legend.Visibility = 1
        legend.Title = (context['source'] + ' ' if context['source'] != 'instantaneous' else '') + labels[4]
        if wall:
            legend.Title = '|mean wall shear|' if context['source']=='reynolds' else '|wall shear|'
        if json.loads(context['definition']).get('fluctuation', False):
            symbol = {'velocity_x':'u', 'velocity_y':'v', 'velocity_z':'w', 'pressure':'p', 'temperature':'T'}[labels[4]]
            legend.Title = symbol + ("''" if context['source']=='favre' else "'") + ' (' + context['source'].title() + ')'
        legend.ComponentTitle = ''
        legend.WindowLocation = 'Upper Right Corner'
        legend.ScalarBarLength, legend.ScalarBarThickness = .45, 16
        legend.TitleFontSize, legend.LabelFontSize = 16, 14
        legend.TitleColor, legend.LabelColor = [0., 0., 0.], [0., 0., 0.]
        item.update(lut=lut, legend=legend, color=labels[4], color_range=settings[22:24],
                    image_size=words[:2], jpeg_quality=words[2], dpi=words[3],
                    write_eps=bool(words[4] & 2),
                    directory=Path(context['directory']), exclusive_lut=True,
                    physical_bounds=bounds if index_surface else None,
                    geometry_empty=False, display_bounds=bounds)

    def execute(self, info, namespace):
        context = json.loads(self.native.astr_insitu_runtime_context().decode('utf-8'))
        for name in list(self.versions):
            if context['catalog'].get(name, {}).get('version') != self.versions[name]:
                self.retire(name, namespace)
        if context['mode'] == 'catalog':
            namespace['retire_resident_inputs']()
            if namespace['os'].environ.get('ASTR_INSITU_MEMORY_TRACE') == '1':
                print(f'ASTR_RUNTIME_VIEWS step={int(info.timestep)} count={len(namespace["render_objects"])}', flush=True)
            return
        self.current = context
        name, step, time = context['id'], int(info.timestep), float(info.time)
        self.versions[name] = context['version']
        dispatch = namespace['dispatch']
        failed = False
        try:
            if context['labels'][0] in ('time_space','probe_spectrum'):
                self.execute_temporal(info,context,namespace)
                return
            if context['labels'][0]=='wall_curve':
                self.execute_wall(info,context,namespace)
                return
            source = namespace['sources']['runtime_product']
            source.UpdatePipeline(time)
            piece = namespace['local_piece'](source.GetClientSideObject().GetOutputDataObject(0))
            if context['labels'][1]=='geometry':
                from runtime_geometry import publish_geometry
                result=publish_geometry(context,piece,namespace,step,time)
                record=dict(id=name,definition_version=context['version'],step=step,time=time,
                    phase='completed_step',source=context['source'],applied_step=context['applied_step'],
                    applied_time=context['applied_time'],rendering_pipeline=namespace['pipeline'],
                    product=result,statistics=context.get('statistics'),
                    geometry_host_bytes=int(namespace['metadata'](piece,'geometry_host_bytes')),
                    geometry_d2h_bytes=int(namespace['metadata'](piece,'geometry_download_bytes')),field_d2h_bytes=0)
                record['processing_backend']=context['processing_backend']
                if context['processing_backend']=='host':
                    record['field_d2h_bytes']=int(namespace['metadata'](piece,'source_field_download_bytes'))
                try:
                    (Path(context['directory'])/f'frame.step{step:012d}.rank{namespace["rank"]:08d}.json').write_text(
                        json.dumps(record,indent=2))
                except OSError as exc:
                    print('ASTR_INSITU_DIAGNOSTIC_WRITE_FAILED '+str(exc),flush=True)
                namespace['frames'].append((step,time))
                return
            points, cells = dispatch.collective_call(namespace['controller'],
                lambda: namespace['update_geometry'](name, piece))
            if namespace['pipeline']=='compatible':
                prefix=namespace['os'].environ.get('ASTR_INSITU_TEST_RUNTIME_COMPATIBLE_PREFIX')
                if prefix and points:
                    import numpy as np
                    from vtkmodules.util.numpy_support import vtk_to_numpy
                    try:
                        np.savez(f'{prefix}.{name}.step{step}.rank{namespace["rank"]}.npz',
                            xyz=vtk_to_numpy(piece.GetPoints().GetData()),
                            quantity=vtk_to_numpy(piece.GetPointData().GetArray('quantity')),
                            connectivity=vtk_to_numpy(piece.GetCells().GetConnectivityArray()),
                            arity=int(namespace['metadata'](piece,'device_arity')))
                    except OSError as exc:
                        print('ASTR_INSITU_DIAGNOSTIC_WRITE_FAILED '+str(exc),flush=True)
            item = namespace['render_objects'][name]
            if context['labels'][4] in ('wall_shear_vector','wall_shear_lines'):
                item['geometry_empty'] = namespace['global_device_bounds'](piece,allow_planar=True,
                    allow_empty=True,minimum_dimensions=0) is None
            if context['labels'][3] != 'grid_index':
                bounds = namespace['global_device_bounds'](piece, allow_planar=True, allow_empty=True)
                item['physical_bounds'] = bounds
                item['geometry_empty'] = bounds is None
            from render_coordinates import update_coordinates, fixed_physical_axis
            coordinate_bounds = item['physical_bounds']
            if coordinate_bounds is None:
                coordinate_bounds = item['display_bounds']
            fixed_axis = None
            if context['labels'][0] == 'slice':
                definition = json.loads(context['definition'])
                fixed_axis = fixed_physical_axis(definition['sampling'], context['reals'][3:6], coordinate_bounds)
            glyph=context['labels'][4]=='wall_shear_vector'
            dispatch.collective_call(namespace['controller'], lambda:
                update_coordinates(item, self.native, context['definition'], coordinate_bounds,
                                   fixed_axis=fixed_axis, empty=item['geometry_empty'],manage_clipping=not glyph))
            if glyph:
                model=item['coordinate_annotation'];clip=list(model['display_bounds'])
                padding=json.loads(context['definition'])['extraction']['length']*max(model['display_scale'])*1.5
                for d in range(3):clip[2*d]-=padding;clip[2*d+1]+=padding
                item['view'].GetClientSideObject().SetMaxClipBounds(clip)
            # A retry can create a new output segment without changing the view definition.
            item['directory'] = Path(context['directory'])
            result = namespace['render_product'].__wrapped__(name, item, step, time, points, cells)
            failed = result['image'] != 'published'
            if failed:
                self.native.astr_insitu_runtime_failure(json.dumps(result['image']).encode('utf-8'))
            record = dict(id=name, definition_version=context['version'], step=step, time=time,
                          processing_backend=context['processing_backend'],
                          definition=json.loads(context['definition']),
                          applied_step=context['applied_step'],applied_time=context['applied_time'],
                          phase='completed_step', source=context['source'],
                          rendering_pipeline=namespace['pipeline'], field_d2h_bytes=0,
                          geometry_host_bytes=(int(namespace['metadata'](piece,'geometry_host_bytes'))
                              if namespace['pipeline']=='compatible' else 0),product=result,
                          geometry_d2h_bytes=(int(namespace['metadata'](piece,'geometry_download_bytes'))
                              if namespace['pipeline']=='compatible' else 0),
                          transfer_scope='selected image fields; explicit compatible geometry/quantity readback and private MPI staging are counted separately',
                          physical_bounds=item['physical_bounds'],
                          geometry_empty=item['geometry_empty'],
                          coordinates=item['coordinate_annotation'],
                          actual_range_scope='local product points; empty ranks have no range',
                          camera=dict(position=list(item['view'].CameraPosition),
                                      focal_point=list(item['view'].CameraFocalPoint),
                                      view_up=list(item['view'].CameraViewUp)))
            if context['processing_backend']=='host':
                record['field_d2h_bytes']=int(namespace['metadata'](piece,'source_field_download_bytes'))
                record['transfer_scope']='explicit host processing: selected source-field download; private MPI staging and formal statistics are counted separately; CPU-generated geometry has no geometry device download'
            if 'statistics' in context:
                record['statistics']=context['statistics']
            if context['labels'][4]=='wall_shear_vector':
                extraction=record['definition']['extraction']
                record['wall_glyph']=dict(count=int(namespace['metadata'](piece,'wall_glyph_count')),
                    length=extraction['length'],stride=extraction['stride'],length_units='native coordinates',
                    amplitude='fixed length; color is norm of the displayed vector',
                    zero_vectors='omitted without a threshold',index_anchor=[0,0],periodic_endpoint='unique node zero',
                    arrowhead='display-compensated; shaft direction and native length are unchanged')
                record['wall_glyph']['metadata_d2h_bytes']=int(namespace['metadata'](piece,'wall_glyph_metadata_d2h_bytes'))
            if context['labels'][4]=='wall_shear_lines':
                keys=('integration_performed','seed_count','particles','control_read_bytes','input_metadata_d2h_bytes',
                      'terminal_length','terminal_steps','terminal_zero','terminal_stage_zero',
                      'terminal_accuracy','terminal_physical_exit','terminal_region_exit')
                record['wall_trace']={key:namespace['metadata'](piece,'wall_trace_'+key) for key in keys}
                record['wall_trace'].update(parameter='physical arc length',field='bilinearly interpolated wall shear',
                    geometry_d2h_bytes=0,termination_scope='global particles; replicated controls',
                    traffic_scope='local bytes for this integration; zero for a reused view')
                record['wall_trace']['final_particles']=[{
                    key:namespace['metadata'](piece,f'wall_trace_final{i}_'+key)
                    for key in ('x','y','z','length','steps','status','direction')}
                    for i in range(int(record['wall_trace']['particles']))]
            if context['labels'][0] == 'streamlines' and context['processing_backend']!='host':
                record['validation_geometry_readback_requested']=bool(
                    namespace['os'].environ.get('ASTR_INSITU_TEST_RUNTIME_LINE_PREFIX'))
                keys=('trajectory_integration_performed','trajectory_seed_count','trajectory_particles',
                      'trajectory_control_read_bytes','trajectory_owner_query_read_bytes',
                      'termination_length','termination_step_budget','termination_domain_exit','termination_subminimum')
                record['streamline']={key:namespace['metadata'](piece,key) for key in keys}
                record['streamline']['termination_scope']='global particles; replicated control metadata'
            try:
                (item['directory'] / f'frame.step{step:012d}.rank{namespace["rank"]:08d}.json').write_text(
                    json.dumps(record, indent=2))
            except OSError as exc:
                print('ASTR_INSITU_DIAGNOSTIC_WRITE_FAILED ' + str(exc), flush=True)
            namespace['frames'].append((step, time))
        except dispatch.ProductFailure as exc:
            failed = True
            self.native.astr_insitu_runtime_failure(str(exc).encode('utf-8'))
        finally:
            namespace['retire_resident_inputs']()
            self.current=None
        if failed:
            self.retire(name, namespace)
        self.current = None

    def execute_temporal(self,info,context,namespace):
        import numpy as np
        from image_publication import publish_pair
        from render_coordinates import native_options,native_unit
        from sbli_products import curve_image_bytes
        from temporal_products import time_space_image_bytes
        dispatch=namespace['dispatch'];step,time=int(info.timestep),float(info.time)
        failed=False
        try:
            def publish():
                if namespace['rank']!=0: return
                function=self.native.astr_insitu_runtime_temporal_data
                function.argtypes=[ctypes.c_int,*[ctypes.POINTER(ctypes.c_int)]*4]
                function.restype=ctypes.c_void_p
                nodes,rows,axis,width=ctypes.c_int(),ctypes.c_int(),ctypes.c_int(),ctypes.c_int()
                def array(which,count):
                    address=function(which,ctypes.byref(nodes),ctypes.byref(rows),ctypes.byref(axis),ctypes.byref(width))
                    scalar={4:ctypes.c_float,8:ctypes.c_double}[width.value]
                    return np.ctypeslib.as_array(ctypes.cast(address,ctypes.POINTER(scalar)),shape=(count(),))
                coordinate=array(0,lambda:nodes.value*(3 if axis.value else 1))
                times=array(1,lambda:rows.value)
                values=array(2,lambda:nodes.value*rows.value).reshape(rows.value,nodes.value)
                gap_function=self.native.astr_insitu_runtime_temporal_gaps
                gap_function.argtypes=[];gap_function.restype=ctypes.POINTER(ctypes.c_int)
                gaps=np.ctypeslib.as_array(gap_function(),shape=(rows.value,))
                words,settings=context['integers'],context['reals']
                common=dict(image_size=words[:2],dpi=words[3],jpeg_quality=words[2],write_eps=bool(words[4]&2))
                if context['labels'][0]=='time_space':
                    flags,scales=native_options(self.native,context['definition'])
                    jpeg,eps=time_space_image_bytes(coordinate.reshape(-1,3),times,values,gaps,axis=axis.value-1,
                        unit=native_unit(self.native),flags=flags,coordinate_values=scales,
                        quantity=context['labels'][4],
                        color_map=context['labels'][6],color_range=settings[22:24],**common)
                else:
                    jpeg,eps=curve_image_bytes(coordinate,values[0],'PSD',x_range=(float(coordinate[0]),float(coordinate[-1])),
                        y_range=settings[22:24],line_width=settings[24],x_label='$f$',**common)
                directory=Path(context['directory'])
                failure=publish_pair(directory/f'frame.step{step:012d}.jpeg',jpeg,eps)
                if failure: raise OSError(failure['errno'],failure['reason'])
                record=dict(id=context['id'],definition_version=context['version'],step=step,time=time,
                    phase='completed_step',definition=json.loads(context['definition']),source=context['source'],
                    rendering_pipeline=namespace['pipeline'],chart_backend='existing-host-matplotlib',
                    transfer_scope='reduced one-dimensional samples only; no volume/final-geometry download',
                    host_temporal_bytes=coordinate.nbytes+times.nbytes+values.nbytes+gaps.nbytes,
                    sample_times=times.tolist(),gaps_before=np.flatnonzero(gaps).tolist(),
                    nodes=nodes.value,rows=rows.value,source_bytes=width.value,
                    product=dict(image_formats=['jpeg']+(['eps'] if eps else []),image_size=words[:2],lighting_enabled=False))
                try:
                    (directory/f'frame.step{step:012d}.rank00000000.json').write_text(json.dumps(record,indent=2))
                except OSError as exc:
                    print('ASTR_INSITU_DIAGNOSTIC_WRITE_FAILED '+str(exc),flush=True)
            dispatch.collective_call(namespace['controller'],publish)
            namespace['frames'].append((step,time))
        except dispatch.ProductFailure as exc:
            failed=True
            self.native.astr_insitu_runtime_failure(str(exc).encode('utf-8'))
        finally:
            if failed: self.retire(context['id'],namespace)
            self.current=None

    def execute_wall(self,info,context,namespace):
        import numpy as np
        from sbli_products import curve_image_bytes
        from image_publication import publish_pair
        dispatch=namespace['dispatch']
        step,time=int(info.timestep),float(info.time)
        failed=False
        try:
            def publish():
                if namespace['rank']!=0:
                    return
                function=self.native.astr_insitu_runtime_wall_data
                function.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)]
                function.restype=ctypes.c_void_p
                count,width,nondimensional=ctypes.c_int(),ctypes.c_int(),ctypes.c_int()
                address=function(ctypes.byref(count),ctypes.byref(width),ctypes.byref(nondimensional))
                scalar={4:ctypes.c_float,8:ctypes.c_double}[width.value]
                data=(np.ctypeslib.as_array(ctypes.cast(address,ctypes.POINTER(scalar)),shape=(4*count.value,)).reshape(-1,4)
                      if count.value else np.empty((0,4)))
                breaks=self.native.astr_insitu_runtime_wall_breaks
                breaks.argtypes=[];breaks.restype=ctypes.POINTER(ctypes.c_ubyte)
                break_before=np.ctypeslib.as_array(breaks(),shape=(count.value,)) if count.value else np.empty(0,dtype=np.uint8)
                symbols={'wall_pressure':'p_w','p_pinf':r'p_w/p_\infty','wall_shear_x':r'\tau_{w,t}',
                         'skin_friction':'C_f','wall_heat_into_gas':r'q_{w,\mathrm{into\ gas}}',
                         'wall_heat_into_wall_normalized':r'\frac{q_{w,\mathrm{into\ wall}}}{\rho_\infty U_\infty^3}'}
                quantity=context['labels'][4];symbol=symbols[quantity]
                if context['source']=='reynolds': symbol=r'\overline{'+symbol+'}'
                units=''
                if not nondimensional.value:
                    units={'wall_pressure':r'\;[\mathrm{Pa}]','wall_shear_x':r'\;[\mathrm{Pa}]',
                           'wall_heat_into_gas':r'\;[\mathrm{W}\,\mathrm{m}^{-2}]'}.get(quantity,'')
                settings,words=context['reals'],context['integers']
                jpeg,eps=curve_image_bytes(data[:,0],data[:,3],'$'+symbol+units+'$',image_size=words[:2],
                    dpi=words[3],jpeg_quality=words[2],x_range=settings[6:8],y_range=settings[22:24],
                    line_width=settings[24],write_eps=bool(words[4]&2),zero_line=quantity=='skin_friction',
                    break_before=break_before,
                    x_label=r'$x/L_\mathrm{ref}$' if nondimensional.value else r'$x\;[\mathrm{m}]$')
                directory=Path(context['directory'])
                failure=publish_pair(directory/f'frame.step{step:012d}.jpeg',jpeg,eps)
                if failure: raise OSError(failure['errno'],failure['reason'])
                record=dict(id=context['id'],definition_version=context['version'],step=step,time=time,
                    phase='completed_step',definition=json.loads(context['definition']),source=context['source'],
                    applied_step=context['applied_step'],applied_time=context['applied_time'],
                    source_bytes=width.value,nondimensional=bool(nondimensional.value),
                    rendering_pipeline=namespace['pipeline'],chart_backend='existing-host-matplotlib',
                    processing_backend=context['processing_backend'],
                    host_curve_bytes=data.nbytes,host_curve_connectivity_bytes=break_before.nbytes,
                    connectivity_breaks=int(np.count_nonzero(break_before)),
                    transfer_scope='shared selected line download is counted in ASTR_RUNTIME_WALL_GROUP',
                    statistics=context.get('statistics'),product=dict(image='published',image_formats=['jpeg']+(['eps'] if eps else []),
                    image_size=words[:2],dpi=words[3],jpeg_quality=words[2],lighting_enabled=False,
                    actual_range=[float(data[:,3].min()),float(data[:,3].max())] if len(data) else None,
                    points=len(data),x_range=settings[6:8],y_range=settings[22:24],line_width=settings[24]))
                try:
                    (directory/f'frame.step{step:012d}.rank00000000.json').write_text(json.dumps(record,indent=2))
                except OSError as exc:
                    print('ASTR_INSITU_DIAGNOSTIC_WRITE_FAILED '+str(exc),flush=True)
            dispatch.collective_call(namespace['controller'],publish)
            namespace['frames'].append((step,time))
        except dispatch.ProductFailure as exc:
            failed=True
            self.native.astr_insitu_runtime_failure(str(exc).encode('utf-8'))
        finally:
            if failed: self.retire(context['id'],namespace)
            self.current=None
