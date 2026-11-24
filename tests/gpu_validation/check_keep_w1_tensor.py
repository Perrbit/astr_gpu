"""Independent x-periodic, y-wall W1 viscous RHS with mixed stresses and energy work."""
import argparse
import json
from pathlib import Path

import numpy as np

from check_keep_w1_scalar import audit, d2_matrix, operators
from check_keep_w1_shear_heat import normal_operators


def test(n, left, inside, shift, frozen=None):
    h, d, e, s, gamma = normal_operators(n, left, inside, shift)
    x = np.arange(n-1)*(2*np.pi/(n-1))
    eta = np.linspace(0., 1., n)
    y = eta+.35*eta*(1-eta)
    jac = 1+.35*(1-2*eta)
    dx, he = 2*np.pi/(n-1), 1/(n-1)
    gy = d/(he*jac[:, None])
    weights = he*h*jac
    ends = [0, -1]
    sign = np.array([-1, 1])
    shape = (n, n-1)

    def gx(f):
        return sum(c*(np.roll(f,-i,axis=1)-np.roll(f,i,axis=1))
                   for i,c in enumerate((.75,-.15,1/60),1))/dx

    def normal(matrix, f):
        return (matrix@f.reshape(n,-1)).reshape(f.shape)

    mu = 1+.2*np.exp(y)
    kappa = .7+.3*np.exp(.6*y)
    viscosity = mu[:, None]*np.array([1,4/3,1])

    def prepare(coefficient):
        b = coefficient/jac
        bw = b[ends]
        flux = bw[:, None]*s/he
        m = -h[:, None]*d2_matrix(b,left,inside)/he + e@flux
        r = m - gy.T@((weights*coefficient)[:, None]*gy)
        diff = flux-(sign*coefficient[ends])[:, None]*gy[ends]
        corr_bound = gamma*bw**2/(he*np.array([b[:6].min(), b[-6:].min()]))
        return r, diff, corr_bound

    ops = [prepare(viscosity[:, i]) for i in range(3)]
    r_temp, diff_temp, bound_temp = prepare(kappa)
    penalty = 2*(2*mu[ends,None]/weights[ends,None] + np.stack([op[2] for op in ops],axis=1))
    penalty_temp = 2*(kappa[ends]/weights[ends]+bound_temp)

    def stress(grad):
        tensor = grad+grad.swapaxes(2,3)
        trace = np.trace(grad,axis1=2,axis2=3)
        tensor -= 2*trace[:,:,None,None]*np.eye(3)/3
        return mu[:,None,None,None]*tensor

    def gradients(u):
        grad = np.zeros((*shape,3,3))
        grad[...,0], grad[...,1] = gx(u), normal(gy,u)
        return grad

    def traction(u, tau):
        wall = sign[:,None,None]*tau[ends,:,:,1]
        for c,(_,diff,_) in enumerate(ops):
            wall[...,c] += diff@u[...,c]
        return wall

    def traction_adjoint(jump):
        material_jump = (sign*mu[ends])[:,None,None]*jump
        # Transpose of the full traction, including the two x/y cross terms.
        rhs = np.einsum('je,exm->jxm',gy[ends].T,material_jump*np.array([1,4/3,1]))
        rhs[ends,:,1] -= gx(material_jump[...,0])
        rhs[ends,:,0] += (2/3)*gx(material_jump[...,1])
        for c,(_,diff,_) in enumerate(ops):
            rhs[...,c] += diff.T@jump[...,c]
        return rhs

    def momentum(u):
        grad = gradients(u)
        tau = stress(grad)
        rhs = gx(weights[:,None,None]*tau[...,0])-normal(gy.T,weights[:,None,None]*tau[...,1])
        for c,(r,_,_) in enumerate(ops):
            rhs[...,c] -= r@u[...,c]
        jump = u[ends]
        wall = traction(u,tau)-penalty[:,None,:]*jump
        rhs[ends] += wall
        rhs += traction_adjoint(jump)
        return rhs, tau, wall

    def energy(u, tau, temp):
        temp_flux = kappa[:,None]*(gy@temp)
        heat_rhs = gx((weights*kappa)[:,None]*gx(temp))-gy.T@(weights[:,None]*temp_flux)-r_temp@temp
        wall_heat = sign[:,None]*temp_flux[ends]+diff_temp@temp-penalty_temp[:,None]*temp[ends]
        heat_rhs += e@wall_heat+gy[ends].T@((sign*kappa[ends])[:,None]*temp[ends])+diff_temp.T@temp[ends]
        pair = (u*u)/2
        work_x, work_y = (np.sum(u*tau[...,i],axis=-1) for i in range(2))
        work_y += np.sum(viscosity[:,None,:]*(normal(gy,pair)-u*normal(gy,u)),axis=-1)
        work = gx(weights[:,None]*work_x)-gy.T@(weights[:,None]*work_y)
        work -= sum(op[0]@pair[...,c] for c,op in enumerate(ops))
        return work+heat_rhs, wall_heat

    amp = np.array([1,.2,.3])
    wave = np.array([1,1,2])
    freq = np.pi*np.array([1,2,1])
    sy, cy = np.sin(y[:,None,None]*freq), np.cos(y[:,None,None]*freq)
    sx, cx = np.sin(x[None,:,None]*wave+[0,np.pi/2,0]), np.cos(x[None,:,None]*wave+[0,np.pi/2,0])
    u = amp*sy*sx
    ux, uy = amp*sy*wave*cx, amp*cy*freq*sx
    uxx, uyy, uxy = -wave**2*u, -freq**2*u, amp*cy*freq*wave*cx
    div = ux[...,0]+uy[...,1]
    graddiv = np.stack([uxx[...,0]+uxy[...,1],uxy[...,0]+uyy[...,1],np.zeros(shape)],axis=-1)
    vgrad = np.stack([ux[...,1],uy[...,1],np.zeros(shape)],axis=-1)
    exact_m = mu[:,None,None]*(uxx+uyy+graddiv/3) + (mu-1)[:,None,None]*(uy+vgrad-(2/3)*div[...,None]*[0,1,0])
    exact_grad = np.stack([ux,uy,np.zeros_like(u)],axis=-1)
    exact_tau = stress(exact_grad)
    rhs, tau, wall_traction = momentum(u)
    perturbed = u + .07*u.max()*np.cos(x[None,:,None]+y[:,None,None]+[0,1,2])
    perturb_rhs, tau_perturbed, perturb_traction = momentum(perturbed)
    kinetic_rate = np.sum(perturb_rhs*perturbed)*dx
    volume_dissipation = dx*(np.sum(weights[:,None,None,None]*gradients(perturbed)*tau_perturbed)
                             + sum(np.sum(perturbed[...,c]*(op[0]@perturbed[...,c])) for c,op in enumerate(ops)))
    if kinetic_rate > -.5*volume_dissipation:
        raise AssertionError('viscous kinetic rate exceeds the borrowing bound for nonzero wall jump')
    jump = np.cos(x[None,:,None]+np.array([0,.7])[:,None,None]+[0,1,2])
    pair = [np.sum(jump*traction(perturbed,tau_perturbed)), np.sum(perturbed*traction_adjoint(jump))]
    adjoint_error = abs(pair[0]-pair[1])/max(1.,abs(pair[0]))
    if adjoint_error > 2.e-12:
        raise AssertionError('mixed traction transpose does not reproduce the wall bilinear form')

    temp = .1*sy[...,0]*np.cos(x[None,:])
    temp_y = .1*np.pi*cy[...,0]*np.cos(x[None,:])
    exact_heat = -(np.pi**2+1)*kappa[:,None]*temp + (.6*(kappa-.7))[:,None]*temp_y
    total_energy, wall_heat = energy(u,tau,temp)
    exact_energy = exact_heat+np.sum(u*exact_m,axis=-1)+np.sum(exact_grad*exact_tau,axis=(2,3))
    energy_balance = abs(dx*(total_energy.sum()-wall_heat.sum()))
    if energy_balance > 3.e-11:
        raise AssertionError('net total-energy change differs from the numerical wall heat')
    err_m = rhs/weights[:,None,None]-exact_m
    err_e = total_energy/weights[:,None]-exact_energy
    if frozen is not None:
        temp_jump = temp+.04*np.cos(x[None,:]+y[:,None])
        energy_jump, heat_jump = energy(perturbed,tau_perturbed,temp_jump)
        expected = np.zeros((*shape,8))
        expected[:,:,:3] = perturb_rhs/weights[:,None,None]
        expected[:,:,3] = energy_jump/weights[:,None]
        expected[ends,:,4:7] = perturb_traction
        expected[ends,:,7] = heat_jump
        frozen.update(h0=h[:6],gamma=gamma,dleft=d[:6,:9],dx=dx,he=he,u=perturbed,t=temp_jump,
                      mu=mu,kappa=kappa,jac=jac,expected=expected)
    return dict(kinetic_rate=kinetic_rate, adjoint_error=adjoint_error, energy_balance=energy_balance, errors=[
        np.max(abs(err_m[:9])), np.max(abs(err_m[9:-9])), np.max(abs(err_e[:9])), np.max(abs(err_e[9:-9])),
        np.max(abs(wall_traction-sign[:,None,None]*exact_tau[ends,:,:,1])),
        np.max(abs(wall_heat-(sign*kappa[ends])[:,None]*temp_y[ends]))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('coefficients',type=Path)
    args = p.parse_args()
    left, inside = operators(args.coefficients)
    shift = audit(args.coefficients)['separate_viscous_d1_shift']
    grids = [25,49,97,193]
    checks = [test(n,left,inside,shift) for n in grids]
    errors = np.array([c['errors'] for c in checks])
    print(json.dumps(dict(nodes=grids, columns=['wall_momentum','interior_momentum','wall_energy','interior_energy',
                                                'wall_traction','wall_heat'], checks=checks,
                          orders=np.log2(errors[:-1]/errors[1:]).tolist()),indent=2))


if __name__ == '__main__':
    main()
