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
#include <atomic>
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

void LoopInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b, Real time, Real dt, int il,
                 int iu, int jl, int ju, int kl, int ku, int ngh);

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Spherical blast wave test problem generator
//========================================================================================

Real hydro_coef = 4;
Real p_amb = 1e-16;
Real rho_amb = 1.35e-3;
Real gamma_hydro = 1.33333333333333333;
Real rin = 1.666666e-3;
// wind

Real k_wind = 1e-6;
Real B_star = 0;
Real r_star = 1e6 / 3e10;

Real R_lc = 0.001;
Real sigma_wind = 10;
Real gamma_wind = 10;

inline void print_par(std::string name, Real value, Real code_val) {
    std::cout << name << " = " << value << ',' << code_val << std::endl;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
    // reading coordinates
    rin = pin->GetOrAddReal("problem", "r_in", 1.6666666666e-3);
    // reading hydro paramters
    gamma_hydro = pin->GetOrAddReal("hydro", "gamma", 1.33333333);
    hydro_coef = gamma_hydro / (gamma_hydro - 1);

    // reading parameters of ambient medium
    p_amb = pin->GetOrAddReal("problem", "p_amb", 1e-15);
    rho_amb = pin->GetOrAddReal("problem", "rho_amb", 1.35e-6);

    B_star = pin->GetOrAddReal("problem", "B_star", 100);
    sigma_wind = pin->GetOrAddReal("problem", "sigma_wind", 10);
    gamma_wind = pin->GetOrAddReal("problem", "gamma_wind", 50);
    Real T = pin->GetOrAddReal("problem", "T", 0.001);

    k_wind = pin->GetOrAddReal("problem", "k_wind", 1e-6);

    /// initializing variables
    Real Omega = 2 * PI / T;

    R_lc = 1 / Omega;

    Real Bphi = -B_star * r_star * r_star / rin / R_lc;

    Real ur = sqrt(gamma_wind * gamma_wind - 1);
    Real rho = (Bphi * Bphi) / (sigma_wind + 1) / gamma_wind / gamma_wind;
    Real p_w = rho * k_wind;

    // ejecta calculations

    const Real uL = 3e10;
    const Real uM = 1.989e33;
    const Real uT = 1;
    const Real uP = 6.66666e22;

    const Real urho = 74;

    const Real uB = 9.15e11;
    const Real uE = 1.8e54;

    print_par("rho_wind", rho * urho, rho);
    print_par("p_wind", p_w, p_w);
    print_par("sig_wind", sigma_wind, sigma_wind);
    print_par("rho_amb", rho_amb * urho, rho_amb);
    print_par("p_amb", p_amb, p_amb);

    if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(BoundaryFace::inner_x1, LoopInnerX1);
    }
    return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
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

    for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
            for (int i = is; i <= ie; i++) {
                phydro->u(IDN, k, j, i) = rho_amb;
                phydro->u(IM1, k, j, i) = 0.0;
                phydro->u(IM2, k, j, i) = 0.0;
                phydro->u(IM3, k, j, i) = 0.0;
                phydro->u(IEN, k, j, i) = (rho_amb + hydro_coef * p_amb) - p_amb;
                ;
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

    for (int k = kl; k <= ku; ++k) {
        for (int j = jl; j <= ju; ++j) {
            for (int i = 1; i <= ngh; ++i) {
                Real r = pcoord->x1v(il - i);
                Real Bphi = -B_star * r_star * r_star / r / R_lc;
                Real ur = sqrt(gamma_wind * gamma_wind - 1);
                Real rho = (Bphi * Bphi) / (sigma_wind + 1) / gamma_wind / gamma_wind;
                prim(IDN, k, j, il - i) = rho;
                prim(IVX, k, j, il - i) = ur;
                prim(IVY, k, j, il - i) = 0.0;
                prim(IVZ, k, j, il - i) = 0.0;
                prim(IPR, k, j, il - i) = rho * k_wind;
            }
        }
    }
    // SET_MAGNETIC_FIELD_BC_OUTFLOW
    for (int k = kl; k <= ku; ++k) {
        for (int j = jl; j <= ju; ++j) {
            for (int i = 1; i <= ngh; ++i) {
                b.x1f(k, j, (il - i)) = 0.0;
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
                Real r = pcoord->x1f(il - i);
                b.x3f(k, j, (il - i)) = -B_star * r_star * r_star / r / R_lc * sigma_wind / (1 + sigma_wind);
            }
        }
    }
}
