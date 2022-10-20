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

void LoopInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b, Real time, Real dt, int il,
                 int iu, int jl, int ju, int kl, int ku, int ngh);

Real t_ej_crit = 0.0;
Real t_ej_end = 0;
Real p_ej = 0.0;
Real p_amb = 0.0;
Real p_jet = 0.0;
Real theta_jet = 0;
Real v_ej = 0;
Real v_jet_r = 0.0;
Real v_jet_phi = 0.0;
Real u_jet = 0.0;

Real t_jet_launch = 0;
Real t_jet_duration = 0;

Real rho_ej = 0;
Real rho_jet = 0;
Real rho_amb = 0;

Real B_r = 0.0;
Real B_phi = 0.0;

Real rin = 1;

Real gamma_ej = 1.0;
Real gamma_jet = 1.0;

void Mesh::InitUserMeshData(ParameterInput *pin) {
    p_amb = pin->GetOrAddReal("problem", "p_amb", 0.0);
    rho_amb = pin->GetOrAddReal("problem", "rho_amb", 0.0);
    rin = pin->GetOrAddReal("problem", "r_in", 1.6e-3);
    Real M_ej = pin->GetOrAddReal("problem", "M_ej", 0.01);

    v_ej = pin->GetOrAddReal("problem", "v_ej", 0.2);
    gamma_ej = 1 / std::sqrt(1 - v_ej * v_ej);
    t_ej_end = pin->GetOrAddReal("problem", "t_ej_duration", 0.015);
    t_ej_crit = pin->GetOrAddReal("problem", "t_ej_crit", 0.005);
    p_ej = pin->GetOrAddReal("problem", "p_ej", 1e4);

    Real int_coef = PI * (2 * t_ej_crit - t_ej_crit * t_ej_crit / t_ej_end);  // integr
    rho_ej = M_ej / (v_ej * rin * rin * int_coef);

    theta_jet = pin->GetOrAddReal("problem", "theta_jet", 0.17453292519943295);
    t_jet_launch = pin->GetOrAddReal("problem", "t_jet_launch", 0.05);
    t_jet_duration = pin->GetOrAddReal("problem", "t_jet_duration", 1);
    v_jet_r = pin->GetOrAddReal("problem", "v_jet_r", 0.8);
    v_jet_phi = pin->GetOrAddReal("problem", "v_jet_phi", 0.4);
    gamma_jet = 1 / std::sqrt(1 - v_jet_r * v_jet_r - v_jet_phi * v_jet_phi);

    Real Gamma_inf = pin->GetOrAddReal("problem", "Gamma_inf", 265);
    Real L_jet = pin->GetOrAddReal("problem", "L_jet", 5e51);

    u_jet = L_jet / (v_jet_r * rin * rin * 4 * PI);

    Real x = u_jet / (Gamma_inf * Gamma_inf);

    B_r = pin->GetOrAddReal("problem", "B_r", 3.1e12);
    B_phi = pin->GetOrAddReal("problem", "B_phi", 6.2e12);

    Real B2 = B_r * B_r + B_phi * B_phi;

    Real sigma_r = 1.0;
    Real sigma_phi = 1.0;
    p_jet = B_r * B_r / (2 * sigma_r);

    rho_jet = 0.5 * (x + sqrt(x * x + 4 * x * (B2 + 4 * p_jet)));

    std::cout << rho_ej << ' ' << rho_jet << std::endl;

    if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(BoundaryFace::inner_x1, LoopInnerX1);
    }

    return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Spherical blast wave test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    Real Gamma = pin->GetOrAddReal("hydro", "gamma", 1.33333333);
    Real Gamma_1 = Gamma - 1.0;
    for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
            for (int i = is; i <= ie; i++) {
                phydro->u(IDN, k, j, i) = rho_amb;
                phydro->u(IM1, k, j, i) = 0.0;
                phydro->u(IM2, k, j, i) = 0.0;
                phydro->u(IM3, k, j, i) = 0.0;
                phydro->u(IEN, k, j, i) = rho_amb + Gamma / Gamma_1 * p_amb - p_amb;
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
    Real rc = 0.043333;
    if (time < t_ej_crit) {
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    Real sin_theta = std::sin(pcoord->x2v(j));
                    prim(IDN, k, j, il - i) = rho_ej * (0.25 + sin_theta * sin_theta * sin_theta);
                    prim(IVX, k, j, il - i) = gamma_ej * v_ej;
                    ;
                    prim(IVY, k, j, il - i) = 0.0;
                    prim(IVZ, k, j, il - i) = 0.0;
                    prim(IPR, k, j, il - i) = p_ej;
                }
            }
        }

        if (MAGNETIC_FIELDS_ENABLED) {
            SET_MAGNETIC_FIELD_BC_ZERO
        }
    } else if (time < t_ej_end) {
        Real vel = v_ej * (1.5 - time / t_ej_crit / 2);
        Real gamma_ej_t = 1 / std::sqrt(1 - vel * vel);
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    Real sin_theta = std::sin(pcoord->x2v(j));
                    prim(IDN, k, j, il - i) =
                        rho_ej * (0.25 + sin_theta * sin_theta * sin_theta) * t_ej_crit * t_ej_crit / time / time;

                    prim(IVX, k, j, il - i) = gamma_ej_t * vel;
                    prim(IVY, k, j, il - i) = 0.0;
                    prim(IVZ, k, j, il - i) = 0.0;
                    prim(IPR, k, j, il - i) = p_ej;
                }
            }
        }
        if (MAGNETIC_FIELDS_ENABLED) {
            SET_MAGNETIC_FIELD_BC_ZERO
        }
    } else {
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    Real sin_theta = std::sin(pcoord->x2v(j));
                    prim(IDN, k, j, il - i) = rho_ej * (0.25 + sin_theta * sin_theta * sin_theta) / 9;
                    prim(IVX, k, j, il - i) = 0;
                    prim(IVY, k, j, il - i) = 0;
                    prim(IVZ, k, j, il - i) = 0;
                    prim(IPR, k, j, il - i) = p_ej;
                }
            }
        }

        if (MAGNETIC_FIELDS_ENABLED) {
            SET_MAGNETIC_FIELD_BC_ZERO
        }
    }

    if ((time >= t_jet_launch + t_ej_end )&& (time < t_jet_launch + t_ej_end + t_jet_duration)) {
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    if (pcoord->x2v(j) < theta_jet) {
                        prim(IDN, k, j, il - i) = rho_jet;
                        prim(IVX, k, j, il - i) = gamma_jet * v_jet_r;
                        prim(IVY, k, j, il - i) = 0.0;
                        prim(IVZ, k, j, il - i) = gamma_jet * v_jet_phi * pcoord->x2v(j) / theta_jet;
                        prim(IPR, k, j, il - i) = p_jet;
                    }
                }
            }
        }

        if (MAGNETIC_FIELDS_ENABLED) {
            for (int k = kl; k <= ku; ++k) {
                for (int j = jl; j <= ju; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        if (pcoord->x2v(j) < theta_jet) {
                            b.x1f(k, j, (il - i)) = B_r;
                        } else {
                            b.x1f(k, j, (il - i)) = 0.0;
                        }
                    }
                }
            }
            for (int k = kl; k <= ku; ++k) {
                for (int j = jl; j <= ju + 1; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        b.x2f(k, j, (il - i)) = 0.0;
                    }
                }
            }
            for (int k = kl; k <= ku + 1; ++k) {
                for (int j = jl; j <= ju; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        if (pcoord->x2v(j) < theta_jet) {
                            Real theta_ratio = pcoord->x2v(j) / theta_jet;
                            b.x3f(k, j, (il - i)) = 2 * B_phi * theta_ratio / (1 + theta_ratio * theta_ratio);
                        } else {
                            b.x3f(k, j, (il - i)) = 0.0;
                        }
                    }
                }
            }
        }
    }

    // copy face-centered magnetic fields into ghost zones
}
