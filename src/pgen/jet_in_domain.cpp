//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file blast.cpp
//! \brief Problem generator for spherical blast wave problem.  Works in Cartesian,
//!        cylindrical, and spherical coordinates.  Contains post-processing code
//!        to check whether blast is spherical for regression tests
//!
//! REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//!   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

// C headers

// C++ headers
#include <math.h>

#include <algorithm>
#include <cmath>
#include <cstdio>   // fopen(), fprintf(), freopen()
#include <cstring>  // strcmp()
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#define SET_MAGNETIC_FIELD_BC_ZERO           \
    for (int k = kl; k <= ku; ++k) {         \
        for (int j = jl; j <= ju; ++j) {     \
            for (int i = 1; i <= ngh; ++i) { \
                b.x1f(k, j, (il - i)) = 0.0; \
            }                                \
        }                                    \
    }                                        \
    for (int k = kl; k <= ku; ++k) {         \
        for (int j = jl; j <= ju + 1; ++j) { \
            for (int i = 1; i <= ngh; ++i) { \
                b.x2f(k, j, (il - i)) = 0.0; \
            }                                \
        }                                    \
    }                                        \
    for (int k = kl; k <= ku + 1; ++k) {     \
        for (int j = jl; j <= ju; ++j) {     \
            for (int i = 1; i <= ngh; ++i) { \
                b.x3f(k, j, (il - i)) = 0.0; \
            }                                \
        }                                    \
    }

#define SET_MAGNETIC_FIELD_BC_OUTFLOW                    \
    for (int k = kl; k <= ku; ++k) {                     \
        for (int j = jl; j <= ju; ++j) {                 \
            for (int i = 1; i <= ngh; ++i) {             \
                b.x1f(k, j, (il - i)) = b.x1f(k, j, il); \
            }                                            \
        }                                                \
    }                                                    \
    for (int k = kl; k <= ku; ++k) {                     \
        for (int j = jl; j <= ju + 1; ++j) {             \
            for (int i = 1; i <= ngh; ++i) {             \
                b.x2f(k, j, (il - i)) = b.x2f(k, j, il); \
            }                                            \
        }                                                \
    }                                                    \
    for (int k = kl; k <= ku + 1; ++k) {                 \
        for (int j = jl; j <= ju; ++j) {                 \
            for (int i = 1; i <= ngh; ++i) {             \
                b.x3f(k, j, (il - i)) = b.x3f(k, j, il); \
            }                                            \
        }                                                \
    }

void LoopInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b, Real time, Real dt, int il,
                 int iu, int jl, int ju, int kl, int ku, int ngh);
//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Spherical blast wave test problem generator
//========================================================================================

Real rin = 0;
Real hydro_coef = 0;
Real rho_amb = 0;
Real keff = 0;
// ejetcta
Real v_ej = 0.2;
Real rho_ej = 0;
Real rc = 0;

// jet
Real t_jet_duration = 1;
Real v_jet_r = 0.8;
Real v_jet_jm = 0.4;
Real rho_jet = 0;

Real B_r = 0;
Real B_jm = 0;

inline void print_par(std::string name, Real value) { std::cout << name << " = " << value << std::endl; }

void Mesh::InitUserMeshData(ParameterInput *pin) {
    rin = pin->GetOrAddReal("problem", "rin", 1.66666666e-3);

    // reading hydro paramters
    Real gamma_hydro = pin->GetOrAddReal("hydro", "gamma", 1.33333333);
    hydro_coef = gamma_hydro / (gamma_hydro - 1);

    // reading parameters of ambient medium
    keff = pin->GetOrAddReal("problem", "keff", 1e-7);
    rho_amb = pin->GetOrAddReal("problem", "rho_amb", 1e-10);

    // reading parameters of ejecta
    rho_ej = pin->GetOrAddReal("problem", "rho_ej", 0.01);
    v_ej = pin->GetOrAddReal("problem", "v_ej", 0.2);
    rc = pin->GetOrAddReal("problem", "r_c", 0.0433333333);

    // reading parameters of jet

    t_jet_duration = pin->GetOrAddReal("problem", "t_jet_duration", 1);
    v_jet_r = pin->GetOrAddReal("problem", "v_jet_r", 0.8);
    v_jet_jm = pin->GetOrAddReal("problem", "v_jet_jm", 0.4);
    rho_jet = pin->GetOrAddReal("problem", "rho_jet", 1e-2);
    B_r = pin->GetOrAddReal("problem", "B_r", 3.388);
    B_jm = pin->GetOrAddReal("problem", "B_jm", 6.777);
    // Real sigma_B = pin->GetOrAddReal("problem", "sigma_B", 1);

    // reading parameters of wind
    Real t_wind_launch = pin->GetOrAddReal("problem", "t_wind_launch", 0.5);
    Real Omega = pin->GetOrAddReal("problem", "Omega", 0.2);
    Real wind_dt = PI / (Omega);
    Real B_wind = pin->GetOrAddReal("problem", "B_mag", 387) * 1e6 / 3e10 / rin;
    Real rho_wind = pin->GetOrAddReal("problem", "rho_wind", 1);

    /// initializing variables

    if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(BoundaryFace::inner_x1, LoopInnerX1);
    }
    return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
            for (int i = is; i <= ie; i++) {
                Real r = pcoord->x1v(i);
                Real sin_theta = pcoord->x2v(j);
                Real rho = 0;
                if (r < rc) {
                    rho = rho_ej * rc * rc / r / r * (0.25 + sin_theta * sin_theta * sin_theta);
                } else if (r <= 4 * rc) {
                    rho = 0.25 * rho_ej * pow(r / rc, -7.0);
                } else {
                    rho = rho_amb;
                }
                Real v = 0;
                if (r < 4 * rc) {
                    v = v_ej * r / rc;
                }
                Real p = keff * pow(rho, 4.0 / 3);

                Real g = 1.0 / sqrt(1 - v * v);
                phydro->u(IDN, k, j, i) = g * rho;
                phydro->u(IM1, k, j, i) = g * g * (rho + hydro_coef * p) * v;
                phydro->u(IM2, k, j, i) = 0.0;
                phydro->u(IM3, k, j, i) = 0.0;
                phydro->u(IEN, k, j, i) = g * g * (rho + hydro_coef * p) - p;
            }
        }
    }

    if (MAGNETIC_FIELDS_ENABLED) {
        for (int k = ks; k <= ke; ++k) {
            for (int j = js; j <= je; ++j) {
                for (int i = is; i <= ie + 1; ++i) {
                    pfield->b.x1f(k, j, i) = 0.0;
                }
            }
        }
        for (int k = ks; k <= ke; ++k) {
            for (int j = js; j <= je + 1; ++j) {
                for (int i = is; i <= ie; ++i) {
                    pfield->b.x2f(k, j, i) = 0.0;
                }
            }
        }
        for (int k = ks; k <= ke + 1; ++k) {
            for (int j = js; j <= je; ++j) {
                for (int i = is; i <= ie; ++i) {
                    pfield->b.x3f(k, j, i) = 0.0;
                }
            }
        }
    }
}

void LoopInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &b, Real time, Real dt, int il,
                 int iu, int jl, int ju, int kl, int ku, int ngh) {
    if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") != 0) {
        std::cout << "use sphereical coordinate" << std::endl;
        exit(0);
    }
    Real theta_jet = 10.0 * PI / 180;
    Real theta_am = 0.2 * theta_jet;

    if (time < t_jet_duration) {
        if (MAGNETIC_FIELDS_ENABLED) {
            for (int k = kl; k <= ku; ++k) {
                for (int j = jl; j <= ju; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        if (pcoord->x2v(j) < theta_jet) {
                            b.x1f(k, j, (il - i)) = B_r;
                        } else {
                            b.x1f(k, j, (il - i)) = b.x1f(k, j, (il));
                        }
                    }
                }
            }
            for (int k = kl; k <= ku; ++k) {
                for (int j = jl; j <= ju + 1; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        if (pcoord->x2v(j) < theta_jet) {
                            b.x2f(k, j, (il - i)) = 0.0;
                        } else {
                            b.x2f(k, j, (il - i)) = b.x2f(k, j, (il));
                        }
                    }
                }
            }
            for (int k = kl; k <= ku + 1; ++k) {
                for (int j = jl; j <= ju; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        Real theta = pcoord->x2v(j);
                        if (theta < theta_jet) {
                            b.x3f(k, j, (il - i)) =
                                2 * B_jm * theta / theta_am / (1 + theta * theta / theta_am / theta_am);
                        } else {
                            b.x3f(k, j, (il - i)) = b.x3f(k, j, (il));
                        }
                    }
                }
            }
        }

        for (int k = kl; k <= ku; ++k) {
            for (int j = ju; j >= jl; --j) {
                for (int i = 1; i <= ngh; ++i) {
                    Real theta = pcoord->x2v(j);
                    if (theta < theta_jet) {
                        Real v_phi = theta / theta_jet * v_jet_jm;
                        Real gamma_jet = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r - v_phi * v_phi);
                        Real tr = theta / theta_jet;
                        Real tr2 = tr * tr;
                        Real tr3 = tr2 * tr;
                        Real tr4 = tr3 * tr;
                        Real tr5 = tr4 * tr;
                        Real index = 11.59617101 * tr5 - 30.90078887 * tr4 + 30.68457504 * tr3 - 10.46822401 * tr2 -
                                     1.28594807 * tr + 6.53961882;
                        Real press = 1.35e-5 * pow(10.0, index);

                        prim(IDN, k, j, il - i) = rho_jet;
                        prim(IVX, k, j, il - i) = gamma_jet * v_jet_r;
                        prim(IVY, k, j, il - i) = 0.0;
                        prim(IVZ, k, j, il - i) = gamma_jet * v_phi;
                        prim(IPR, k, j, il - i) = press;
                    } else {
                        prim(IDN, k, j, il - i) = prim(IDN, k, j, il);
                        prim(IVX, k, j, il - i) = prim(IVX, k, j, il);
                        prim(IVY, k, j, il - i) = prim(IVY, k, j, il);
                        prim(IVZ, k, j, il - i) = prim(IVZ, k, j, il);
                        prim(IPR, k, j, il - i) = prim(IPR, k, j, il);
                    }
                }
            }
        }
    } else {
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    prim(IDN, k, j, il - i) = prim(IDN, k, j, il);
                    prim(IVX, k, j, il - i) = prim(IVX, k, j, il);
                    prim(IVY, k, j, il - i) = prim(IVY, k, j, il);
                    prim(IVZ, k, j, il - i) = prim(IVZ, k, j, il);
                    prim(IPR, k, j, il - i) = prim(IPR, k, j, il);
                }
            }
        }
        if (MAGNETIC_FIELDS_ENABLED) {
            SET_MAGNETIC_FIELD_BC_OUTFLOW
        }
    }

    // copy face-centered magnetic fields into ghost zones
}
