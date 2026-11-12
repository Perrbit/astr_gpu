"""Read-only physical-CURVE metric audit; do not admit or advance a flow."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np

from check_curvilinear_metrics import read_astr_metrics
from generate_curvilinear_tgv_grid import mapped_grid
from run_keep_metric_audit import periodic_derivative
from run_keep_simple_boundary_gate import prepare


def metric_errors(jacobian,inverse,scheme,matrix,amplitude):
    n=jacobian.shape[0]-1
    if jacobian.shape!=(n+1,)*3 or matrix.shape!=(n+1,n+1):
        raise AssertionError('Physical metric audit requires the specified cubic derivative matrix')
    if not np.isfinite(inverse).all() or not np.isfinite(jacobian).all() or np.min(jacobian)<=0:
        raise AssertionError('Physical CURVE metrics are nonfinite or have nonpositive Jacobian')
    h=2*np.pi/n
    xi,eta,zeta=np.meshgrid(*([np.linspace(0,2*np.pi,n+1)]*3),indexing='ij')
    basis=np.broadcast_to(np.eye(3),jacobian.shape+(3,3)).copy()
    basis[...,1,0]=amplitude*np.cos(xi)*np.sin(zeta)
    basis[...,1,2]=amplitude*np.sin(xi)*np.cos(zeta)
    exact_inverse=np.linalg.inv(basis)
    area=(jacobian[...,None,None]*inverse)[:-1,:,:-1]
    residual=periodic_derivative(area[...,0,:],0,scheme)
    residual+=np.einsum('jl,ilkb->ijkb',matrix,area[...,1,:])
    residual+=periodic_derivative(area[...,2,:],2,scheme)
    # Physical y faces have separate boundary RHS; inspect only volume rows.
    residual=residual[:,1:-1]
    pressure_rhs=-residual/jacobian[:-1,1:-1,:-1,None]
    scalar=eta+amplitude*np.sin(xi)*np.sin(zeta)
    gradient=sum(inverse[...,axis,:]*derivative(scalar,axis,scheme,matrix)[...,None] for axis in range(3))
    affine_error=abs(gradient-np.array([0.,1.,0.]))
    report=dict(jacobian_min=float(jacobian.min()),
        analytic_scaled_jacobian_max_abs=float(np.max(abs(jacobian/h**3-1))),
        analytic_scaled_inverse_max_abs=float(np.max(abs(inverse*h-exact_inverse))),
        metric_identity_max_abs=float(np.max(abs(residual))),
        stagnant_pressure_rhs_max_abs=float(np.max(abs(pressure_rhs))),
        physical_affine_y_gradient_max_abs=float(affine_error.max()),
        physical_affine_y_gradient_volume_max_abs=float(affine_error[:,1:-1].max()),
        physical_volume_y_rows=[1,n-1])
    return report


def derivative(field,axis,scheme,matrix):
    if axis==1:
        return np.moveaxis(np.tensordot(matrix,field,axes=(1,axis)),0,axis)
    unique=np.take(field,np.arange(field.shape[axis]-1),axis=axis)
    values=periodic_derivative(unique,axis,scheme)
    return np.concatenate((values,np.take(values,[0],axis=axis)),axis=axis)


def mixed_curl_candidate(coordinates,scheme,matrix,jacobian):
    """Evaluate an unapproved geometry candidate only; never replace solver metrics."""
    n=coordinates.shape[0]-1
    origin=coordinates[0,0,0]
    background=np.column_stack([(coordinates[n,0,0]-origin)/n,
        (coordinates[0,n,0]-origin)/n,(coordinates[0,0,n]-origin)/n])
    indices=np.moveaxis(np.indices((n+1,)*3),0,-1)
    displacement=coordinates-origin-np.einsum('ab,...b->...a',background,indices)
    dw=[derivative(displacement,axis,scheme,matrix) for axis in range(3)]
    area=np.empty(jacobian.shape+(3,3))
    for axis in range(3):
        beta=(axis+1)%3;gamma=(axis+2)%3
        area[...,axis,:]=np.cross(background[:,beta],background[:,gamma])+np.cross(background[:,beta],dw[gamma])
        area[...,axis,:]+=np.cross(dw[beta],background[:,gamma])
        area[...,axis,:]+=.5*(derivative(np.cross(displacement,dw[gamma]),beta,scheme,matrix)
            -derivative(np.cross(displacement,dw[beta]),gamma,scheme,matrix))
    return jacobian,area/jacobian[...,None,None]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','mpiexec','matrix-directory','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    opts=parser.parse_args()
    out=opts.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',OMP_NUM_THREADS='1',
        ASTR_GEOMETRY_DUMP='geometry.h5')
    report=dict(status='running',scope='CPU geometry only; zero complete steps; not CURVE boundary admission',
                mapping='y=eta+0.15*sin(xi)*sin(zeta), x=xi, z=zeta',
                boundaries=[1,1,41,41,1,1],derivative_matrix_source=str(opts.matrix_directory.resolve()),cases={})
    for scheme,n in (('643e',16),('643c',16),('643e',32),('643c',32)):
        baseline=None
        rows=np.loadtxt(opts.matrix_directory/f'matrices_n{n}.txt')
        matrix=np.zeros((n+1,n+1))
        matrix[rows[:,0].astype(int),rows[:,1].astype(int)]=rows[:,2 if scheme=='643e' else 3]
        topologies=((1,1,1),(2,1,1),(1,2,1),(1,1,2)) if n==16 else ((1,1,1),)
        for topology in topologies:
            ranks=int(np.prod(topology));label=scheme+f'_n{n}_'+''.join(map(str,topology))
            case=out/label
            prepare(case,'41y','cpu',scheme,'643e','off',steps=0,checkpoint=False,intervals=n)
            x,y,z,_=mapped_grid(n,n,n,.15,'y-wavy')
            with h5py.File(case/'datin/grid.simple.h5','w') as grid:
                for name,values in zip(('x','y','z'),(x,y,z)):
                    grid[name]=values.transpose(2,1,0)
            command=[str(opts.mpiexec.absolute()),'--mca','coll_hcoll_enable','0','-np',str(ranks),
                     str(opts.cpu.resolve()),'run','datin/input.case']
            local=dict(env,ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)))
            with (case/'run.log').open('w') as log:
                subprocess.run(command,cwd=case,env=local,stdout=log,stderr=subprocess.STDOUT,check=True)
            jacobian,inverse=read_astr_metrics(case/'geometry.h5')
            entry=metric_errors(jacobian,inverse,scheme,matrix,.15)
            entry.update(command=command,topology=list(topology))
            if topology==(1,1,1):
                candidate_j,candidate_inverse=mixed_curl_candidate(np.stack((x,y,z),axis=-1),scheme,matrix,jacobian)
                entry['unapproved_mixed_curl_candidate']=metric_errors(candidate_j,candidate_inverse,scheme,matrix,.15)
            if baseline is None:
                baseline=(jacobian,inverse)
            else:
                entry['decomposition_max_scaled']=max(float(np.max(abs(a-b)/np.maximum(1,np.maximum(abs(a),abs(b)))))
                    for a,b in zip(baseline,(jacobian,inverse)))
            report['cases'][label]=entry
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(label,entry,flush=True)
            if entry['metric_identity_max_abs']>2e-10 or entry['stagnant_pressure_rhs_max_abs']>2e-10:
                raise AssertionError('Physical CURVE metric/free-stream gate failed; do not advance')
            if entry.get('decomposition_max_scaled',0)>2e-10:
                raise AssertionError('Physical CURVE geometry depends on MPI decomposition')
    report['status']='metric_identity_and_MPI_measured; analytic_accuracy_not_accepted; no_boundary_admission'
    for scheme in ('643e','643c'):
        entries=[report['cases'][scheme+f'_n{n}_111'] for n in (16,32)]
        report.setdefault('inverse_error_orders',{})[scheme]=dict(
            production=float(np.log2(entries[0]['analytic_scaled_inverse_max_abs']/entries[1]['analytic_scaled_inverse_max_abs'])),
            unapproved_candidate=float(np.log2(entries[0]['unapproved_mixed_curl_candidate']['analytic_scaled_inverse_max_abs']
                /entries[1]['unapproved_mixed_curl_candidate']['analytic_scaled_inverse_max_abs'])))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
