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

Real hydro_coef = 4;
Real K_amb = 4e-6;
Real rho_amb = 0;
Real gamma_hydro = 1.33333333333333333;

// ejetcta
Real t_ej_crit = 0.005;
Real t_ej_end = 0.015;
Real v_ej = 0.2;
Real rho_ej = 0;
Real K_ej = 1e-4;

// jet
Real theta_jet = 0.1;
Real t_jet_launch = 0.5;
Real t_jet_duration = 1;
Real v_jet_r = 0.8;
Real v_jet_jm = 0.4;
Real rho_jet = 0;
Real p_jet = 0;

Real B_r = 0;
Real B_jm = 0;

inline void print_par(std::string name, Real value) { std::cout << name << " = " << value << std::endl; }

void Mesh::InitUserMeshData(ParameterInput *pin) {
    // reading coordinates
    Real rin = pin->GetOrAddReal("problem", "r_in", 1.6666666666e-3);
    // reading hydro paramters
    gamma_hydro = pin->GetOrAddReal("hydro", "gamma", 1.33333333);
    Real hydro_coef = gamma_hydro / (gamma_hydro - 1);

    // reading parameters of ambient medium
    K_amb = pin->GetOrAddReal("problem", "K_amb", 4e-6);
    rho_amb = pin->GetOrAddReal("problem", "rho_amb", 1.35e-6);

    // reading parameters of ejecta
    K_ej = pin->GetOrAddReal("problem", "K_ej", 4e-6);
    Real M_ej = pin->GetOrAddReal("problem", "M_ej", 0.01);
    t_ej_crit = pin->GetOrAddReal("problem", "t_ej_crit", 0.005);
    t_ej_end = pin->GetOrAddReal("problem", "t_ej_duration", 0.015);
    v_ej = pin->GetOrAddReal("problem", "v_ej", 0.2);

    // reading parameters of jet
    theta_jet = pin->GetOrAddReal("problem", "theta_jet", 0.17453292519943295);
    t_jet_launch = pin->GetOrAddReal("problem", "t_jet_launch", 0.01);
    t_jet_duration = pin->GetOrAddReal("problem", "t_jet_duration", 1);
    v_jet_r = pin->GetOrAddReal("problem", "v_jet_r", 0.8);
    v_jet_jm = pin->GetOrAddReal("problem", "v_jet_jm", 0.4);
    Real Gamma_inf = pin->GetOrAddReal("problem", "Gamma_inf", 300);
    Real L_jet = pin->GetOrAddReal("problem", "L_jet", 0.00278);
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
    // ejecta calculations
    Real int_coef = 2 * PI * (0.5 + 3.0 / 8 * PI) * (2 * t_ej_crit - t_ej_crit * t_ej_crit / t_ej_end);

    rho_ej = M_ej / (v_ej * rin * rin * int_coef);

    // jet calculations
    Real gamma_jet_r = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r);  // gamma in radial direction

    Real w = Gamma_inf / gamma_jet_r;

    Real e_jet = L_jet / (v_jet_r * rin * rin * 4 * PI);

    rho_jet = e_jet / gamma_jet_r / gamma_jet_r / w;

    Real b2 = 1;

    Real sigma = b2 / (4 * PI * w);

    Real eta = w / (1 + sigma);

    p_jet= rho_jet * (eta - 1) * (gamma_hydro - 1) / gamma_hydro;

    print_par("rho_ej", rho_ej);
    print_par("rho_jet", rho_jet);
    print_par("sigma", sigma);
    print_par("p_jet", p_jet);
    print_par("w", w);
    print_par("eta", eta);

    if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(BoundaryFace::inner_x1, LoopInnerX1);
    }
    return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    Real p_amb = K_amb * pow(rho_amb, gamma_hydro);

    for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
            for (int i = is; i <= ie; i++) {
                phydro->u(IDN, k, j, i) = rho_amb;
                phydro->u(IM1, k, j, i) = 0.0;
                phydro->u(IM2, k, j, i) = 0.0;
                phydro->u(IM3, k, j, i) = 0.0;
                phydro->u(IEN, k, j, i) = (rho_amb + hydro_coef * p_amb) - p_amb;
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

    if (time < t_ej_crit) {
        Real gamma_ej = 1 / std::sqrt(1 - v_ej * v_ej);
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    Real sin_theta = std::sin(pcoord->x2v(j));
                    Real rho = rho_ej * (0.25 + sin_theta * sin_theta * sin_theta);
                    prim(IDN, k, j, il - i) = rho;
                    prim(IVX, k, j, il - i) = gamma_ej * v_ej;
                    prim(IVY, k, j, il - i) = 0.0;
                    prim(IVZ, k, j, il - i) = 0.0;
                    prim(IPR, k, j, il - i) = K_ej * pow(rho, gamma_hydro);
                }
            }
        }

        if (MAGNETIC_FIELDS_ENABLED) {
            SET_MAGNETIC_FIELD_BC_OUTFLOW
        }
    } else if (time <= t_ej_end) {
        Real vel = v_ej * (t_ej_crit * t_ej_crit * t_ej_crit * t_ej_crit) / (time * time * time * time);
        Real gamma_ej = 1 / std::sqrt(1 - vel * vel);
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    Real sin_theta = std::sin(pcoord->x2v(j));
                    Real rho =
                        rho_ej * (0.25 + sin_theta * sin_theta * sin_theta) * t_ej_crit * t_ej_crit / time / time;
                    prim(IDN, k, j, il - i) = rho;
                    prim(IVX, k, j, il - i) = gamma_ej * vel;
                    prim(IVY, k, j, il - i) = 0.0;
                    prim(IVZ, k, j, il - i) = 0.0;
                    prim(IPR, k, j, il - i) = K_ej * pow(rho, gamma_hydro);
                }
            }
        }
        if (MAGNETIC_FIELDS_ENABLED) {
            SET_MAGNETIC_FIELD_BC_OUTFLOW
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

    if ((time >= t_jet_launch) && (time < t_jet_launch + t_jet_duration)) {
        if (MAGNETIC_FIELDS_ENABLED) {
            for (int k = kl; k <= ku; ++k) {
                for (int j = jl; j <= ju; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        if (pcoord->x2v(j) < theta_jet) {
                            b.x1f(k, j, (il - i)) = B_r;
                        } else {
                            b.x1f(k, j, (il - i)) = b.x1f(k, j, il);
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
                            b.x2f(k, j, (il - i)) = b.x2f(k, j, il);
                        }
                    }
                }
            }
            for (int k = kl; k <= ku + 1; ++k) {
                for (int j = jl; j <= ju; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        if (pcoord->x2v(j) < theta_jet) {
                            b.x3f(k, j, (il - i)) = B_jm;
                        } else {
                            b.x3f(k, j, (il - i)) = b.x3f(k, j, il);
                        }
                    }
                }
            }
        }

        for (int k = kl; k <= ku; ++k) {
            for (int j = ju; j >= jl; --j) {
                for (int i = 1; i <= ngh; ++i) {
                    if (pcoord->x2v(j) < theta_jet) {
                        Real v_phi = pcoord->x2v(j) / theta_jet * v_jet_jm;
                        Real gamma_jet = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r - v_phi * v_phi);

                        prim(IDN, k, j, il - i) = rho_jet;
                        prim(IVX, k, j, il - i) = gamma_jet * v_jet_r;
                        prim(IVY, k, j, il - i) = 0.0;
                        prim(IVZ, k, j, il - i) = gamma_jet * v_phi;
                        prim(IPR, k, j, il - i) = p_jet;
                    } else if (time > t_ej_end) {
                        prim(IDN, k, j, il - i) = prim(IDN, k, j, il);
                        prim(IVX, k, j, il - i) = prim(IVX, k, j, il);
                        prim(IVY, k, j, il - i) = prim(IVY, k, j, il);
                        prim(IVZ, k, j, il - i) = prim(IVZ, k, j, il);
                        prim(IPR, k, j, il - i) = prim(IPR, k, j, il);
                    }
                }
            }
        }
    } else if (time >= t_jet_launch + t_jet_duration) {
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

    /*if (time >= t_wind_launch) {
        int index = int(time / wind_dt);
        // std::cout << "index = " << index << std::endl;
        Real B_sign = 1.0;
        // index % 2 == 0 ? 1.0 : -1.0;
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    prim(IDN, k, j, il - i) = rho_wind;
                    prim(IVX, k, j, il - i) = 0.1 / sqrt(1 - 0.01);
                    prim(IVY, k, j, il - i) = 0.0;
                    prim(IVZ, k, j, il - i) = 0.0;
                    prim(IPR, k, j, il - i) = B_wind * B_wind / 2;
                }
            }
        }

        if (MAGNETIC_FIELDS_ENABLED) {
            for (int k = kl; k <= ku; ++k) {
                for (int j = jl; j <= ju; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        b.x1f(k, j, (il - i)) = b.x1f(k, j, (il));
                    }
                }
            }
            for (int k = kl; k <= ku; ++k) {
                for (int j = jl; j <= ju + 1; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        b.x2f(k, j, (il - i)) = b.x2f(k, j, (il));
                    }
                }
            }
            for (int k = kl; k <= ku + 1; ++k) {
                for (int j = jl; j <= ju; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        b.x3f(k, j, (il - i)) = B_sign * B_wind;
                    }
                }
            }
        }
    }*/

    // copy face-centered magnetic fields into ghost zones
}
