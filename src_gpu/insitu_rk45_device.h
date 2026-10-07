// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
// FP64 device adaptation of VTK 9.6.2 Common/Math/vtkRungeKutta45.cxx.
// The evaluator supplies the normalized tangent for length-unit integration.
#ifndef ASTR_INSITU_RK45_DEVICE_H
#define ASTR_INSITU_RK45_DEVICE_H

#include <cmath>

#ifdef __CUDACC__
#define ASTR_RK45_EXEC __host__ __device__
#else
#define ASTR_RK45_EXEC
#endif

namespace astr_insitu {
enum RK45Status { StepOk = 0, Outside = 1, Unexpected = 3, Nonfinite = 4 };
enum TraceStatus { TraceActive = 0, TraceLength = 10, TraceSteps = 11, TraceTransfer = 12 };

struct TraceState {
  double point[3], suggested, length, error;
  int steps, status;
};

// VTK rejects the final request when remaining propagation is below minStep.
// Classify that exact pre-step return; do not extrapolate or accept other errors.
ASTR_RK45_EXEC inline bool trace_subminimum_stop(const TraceState& state,
    double target,double minimum)
{
  const double remaining=target-state.length;
  return state.status==Unexpected && std::isfinite(target) && std::isfinite(minimum) &&
    minimum>0. && std::isfinite(state.length) && state.length>=0. &&
    remaining>0. && remaining<minimum && std::isfinite(state.suggested) &&
    std::abs(state.suggested)==remaining && state.error==1.e299;
}

template <class Evaluator>
ASTR_RK45_EXEC int cash_karp_step(const Evaluator& evaluate, const double* previous,
    double* next, double parameter, double step, double& actual, double& error)
{
  const double a[5] = {1./5., 3./10., 3./5., 1., 7./8.};
  const double b[5][5] = {
    {1./5., 0., 0., 0., 0.},
    {3./40., 9./40., 0., 0., 0.},
    {3./10., -9./10., 6./5., 0., 0.},
    {-11./54., 5./2., -70./27., 35./27., 0.},
    {1631./55296., 175./512., 575./13824., 44275./110592., 253./4096.}
  };
  const double c[6] = {37./378., 0., 250./621., 125./594., 0., 512./1771.};
  const double dc[6] = {37./378.-2825./27648., 0., 250./621.-18575./48384.,
    125./594.-13525./55296., -277./14336., 512./1771.-1./4.};
  double k[6][3], point[3];
  actual = 0.;
  for (int d = 0; d < 3; ++d) point[d] = previous[d];
  if (!evaluate(point, parameter, k[0])) {
    for (int d = 0; d < 3; ++d) next[d] = point[d];
    return Outside;
  }
  for (int stage = 1; stage < 6; ++stage) {
    for (int d = 0; d < 3; ++d) {
      double sum = 0.;
      for (int j = 0; j < stage; ++j) sum += b[stage-1][j] * k[j][d];
      point[d] = previous[d] + step * sum;
      if (!std::isfinite(point[d])) return Nonfinite;
    }
    if (!evaluate(point, parameter + step*a[stage-1], k[stage])) {
      for (int d = 0; d < 3; ++d) next[d] = point[d];
      actual = step*a[stage-1];
      return Outside;
    }
  }
  int unchanged = 0;
  double err = 0.;
  for (int d = 0; d < 3; ++d) {
    double sum = 0.;
    for (int j = 0; j < 6; ++j) {
      if (!std::isfinite(k[j][d])) return Nonfinite;
      sum += c[j] * k[j][d];
    }
    next[d] = previous[d] + step * sum;
    if (!std::isfinite(next[d])) return Nonfinite;
    unchanged += next[d] == previous[d];
    sum = 0.;
    for (int j = 0; j < 6; ++j) sum += dc[j] * k[j][d];
    err += step * sum * step * sum;
  }
  actual = step;
  error = std::sqrt(err);
  if (!std::isfinite(error)) return Nonfinite;
  return unchanged == 3 ? Unexpected : StepOk;
}

template <class Evaluator>
ASTR_RK45_EXEC int rk45_step(const Evaluator& evaluate, const double* previous,
    double* next, double parameter, double& step, double& actual,
    double minimum, double maximum, double tolerance, double& error)
{
  error = 1.e299; // VTK_DOUBLE_MAX, also returned before an out-of-domain step.
  actual = 0.;
  if (!std::isfinite(parameter) || !std::isfinite(step) || step == 0. ||
      !std::isfinite(minimum) || !std::isfinite(maximum) ||
      !std::isfinite(tolerance)) return Nonfinite;
  for (int d = 0; d < 3; ++d)
    if (!std::isfinite(previous[d])) return Nonfinite;
  minimum = std::abs(minimum);
  maximum = std::abs(maximum);
  const double magnitude = std::abs(step);
  if ((minimum == magnitude && maximum == magnitude) || tolerance <= 0.)
    return cash_karp_step(evaluate, previous, next, parameter, step, actual, error);
  if (minimum > maximum || minimum == 0.) return Unexpected;

  while (error > tolerance) {
    int status = cash_karp_step(evaluate, previous, next, parameter, step, actual, error);
    if (status != StepOk) return status;
    if (std::abs(step) == minimum) break;
    const double ratio = error / tolerance;
    double candidate;
    if (ratio == 0.) candidate = step < 0. ? -minimum : minimum;
    else candidate = .9 * step * std::pow(ratio, ratio > 1. ? -.25 : -.2);
    if (!std::isfinite(candidate)) return Nonfinite;
    bool bound = false;
    if (std::abs(candidate) > maximum) {
      step = maximum * step / std::abs(step);
      bound = true;
    } else if (std::abs(candidate) < minimum) {
      step = minimum * step / std::abs(step);
      bound = true;
    } else step = candidate;
    if (parameter + step == parameter) return Unexpected;
    if (bound)
      return cash_karp_step(evaluate, previous, next, parameter, step, actual, error);
  }
  return StepOk;
}

// A bounded geometry segment; the caller routes only the small particle state
// when ownership changes. Geometry includes the initial point for seam joining.
template <class Evaluator, class Owner, class Writer>
ASTR_RK45_EXEC int trace_segment(const Evaluator& evaluate, const Owner& owner,
    const Writer& write, TraceState& state, int limit, double minimum,
    double maximum, double tolerance, double target, int maximum_steps)
{
  if (state.status != TraceActive) return 0;
  if (!std::isfinite(target) || target <= 0. || !std::isfinite(state.length) ||
      state.length < 0. || limit <= 0 || maximum_steps <= 0 || state.steps < 0) {
    state.status = Nonfinite;
    return 0;
  }
  for (int d=0;d<3;++d) if (!std::isfinite(state.point[d])) {
    state.status = Nonfinite;
    return 0;
  }
  int vertices=1;
  write(0,state.point,state.length,0.);
#ifdef __CUDACC__
#pragma unroll 1
#endif
  for (int iteration=0;iteration<limit;++iteration) {
    if (state.length >= target) { state.status=TraceLength; break; }
    if (state.steps >= maximum_steps) { state.status=TraceSteps; break; }
    if (!owner(state.point)) { state.status=TraceTransfer; break; }
    double request=state.suggested, bound=maximum;
    const double remaining=target-state.length;
    // Adaptation may enlarge the current request: cap the whole call, not only its input.
    if (bound>remaining) bound=remaining;
    if (std::abs(request)>remaining) {
      request=std::copysign(remaining,request);
    }
    double next[3]={state.point[0],state.point[1],state.point[2]},actual=0.;
    const int status=rk45_step(evaluate,state.point,next,0.,request,actual,
                               minimum,bound,tolerance,state.error);
    state.suggested=request;
    if (status!=StepOk) { state.status=status; break; }
    // Use accepted length, never the NEXT suggested step returned by RK45.
    state.length+=std::abs(actual);
    if (!std::isfinite(state.length)) { state.status=Nonfinite; break; }
    for (int d=0;d<3;++d) state.point[d]=next[d];
    ++state.steps;
    write(vertices++,state.point,state.length,actual);
  }
  // A full final chunk must expose termination/handoff without another launch.
  if (state.status==TraceActive) {
    if (state.length>=target) state.status=TraceLength;
    else if (state.steps>=maximum_steps) state.status=TraceSteps;
    else if (!owner(state.point)) state.status=TraceTransfer;
  }
  return vertices;
}
} // namespace astr_insitu
#undef ASTR_RK45_EXEC
#endif
