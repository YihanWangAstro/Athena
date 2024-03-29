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

#define SET_MAGNETIC_FIELD_BC_REFLECTING                          \
    for (int k = kl; k <= ku; ++k) {                              \
        for (int j = jl; j <= ju; ++j) {                          \
            for (int i = 1; i <= ngh; ++i) {                      \
                b.x1f(k, j, (il - i)) = -b.x1f(k, j, il + i - 1); \
            }                                                     \
        }                                                         \
    }                                                             \
    for (int k = kl; k <= ku; ++k) {                              \
        for (int j = jl; j <= ju + 1; ++j) {                      \
            for (int i = 1; i <= ngh; ++i) {                      \
                b.x2f(k, j, (il - i)) = b.x2f(k, j, il + i - 1);  \
            }                                                     \
        }                                                         \
    }                                                             \
    for (int k = kl; k <= ku + 1; ++k) {                          \
        for (int j = jl; j <= ju; ++j) {                          \
            for (int i = 1; i <= ngh; ++i) {                      \
                b.x3f(k, j, (il - i)) = b.x3f(k, j, il + i - 1);  \
            }                                                     \
        }                                                         \
    }

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

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Spherical blast wave test problem generator
//========================================================================================

Real hydro_coef = 4;
Real p_amb = 1e-16;
Real rho_amb = 1.35e-3;
Real gamma_hydro = 1.33333333333333333;
Real rin = 1.666666e-3;
// ejetcta
Real v_ej = 0.2;
Real rho_ej = 0;
Real rho_tail = 0;
Real k_ej = 1e-6;
Real r_c = 0.04333333;
Real tail_n = 8;
bool ej_on = true;

// jet
Real theta_jet = 0.1;
Real t_jet_duration = 1;
Real v_jet_r = 0.8;
Real v_jet_jm = 0.4;
Real rho_jet = 0;
Real p_jet = 0;
Real B_jm = 0;
bool jet_on = true;
// wind
Real t_wind_launch = 0;
Real t_wind_last = 10;
Real B_wind = 0;
Real rho_wind = 0;
Real v_wind = 0.1;
Real p_wind = 0;

inline void print_par(std::string name, Real value, Real code_val) {
    std::cout << name << " = " << value << ',' << code_val << std::endl;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
    // reading model parameters

    // reading coordinates
    rin = pin->GetOrAddReal("problem", "r_in", 1.6666666666e-3);
    // reading hydro paramters
    gamma_hydro = pin->GetOrAddReal("hydro", "gamma", 1.33333333);
    hydro_coef = gamma_hydro / (gamma_hydro - 1);

    // reading parameters of ambient medium
    p_amb = pin->GetOrAddReal("problem", "p_amb", 1e-15);
    rho_amb = pin->GetOrAddReal("problem", "rho_amb", 1.35e-6);

    // reading parameters of ejecta
    ej_on = pin->GetBoolean("problem", "ej_on");
    k_ej = pin->GetOrAddReal("problem", "k_ej", 1e-7);
    Real M_ej = pin->GetOrAddReal("problem", "M_ej", 0.01);
    v_ej = pin->GetOrAddReal("problem", "v_ej", 0.2);
    r_c = pin->GetOrAddReal("problem", "r_c", 0.043333);

    // reading parameters of jet
    jet_on = pin->GetBoolean("problem", "jet_on");
    theta_jet = pin->GetOrAddReal("problem", "theta_jet", 0.17453292519943295);
    t_jet_duration = pin->GetOrAddReal("problem", "t_jet_duration", 1);
    v_jet_r = pin->GetOrAddReal("problem", "v_jet_r", 0.8);
    v_jet_jm = pin->GetOrAddReal("problem", "v_jet_jm", 0.4);
    Real Gamma_inf = pin->GetOrAddReal("problem", "Gamma_inf", 300);
    Real L_jet = pin->GetOrAddReal("problem", "L_jet", 0.00278);
    Real sigma_jet = pin->GetOrAddReal("problem", "sigma_jet", 1);

    // B_jm = pin->GetOrAddReal("problem", "B_jm", 6.777);
    //  Real sigma_B = pin->GetOrAddReal("problem", "sigma_B", 1);

    // reading parameters of wind
    t_wind_launch = pin->GetOrAddReal("problem", "t_wind_launch", 6);
    t_wind_last = pin->GetOrAddReal("problem", "t_wind_last", 10);
    Real E_wind = pin->GetOrAddReal("problem", "E_wind", 6.66666e-6);
    Real sigma_wind = pin->GetOrAddReal("problem", "sigma_wind", 10);
    Real k_wind = pin->GetOrAddReal("problem", "k_wind", 1e-2);

    /// initializing variables
    Real gamma_wind = 100;

    v_wind = sqrt(1 - 1 / gamma_wind / gamma_wind);

    Real L_wind = E_wind / t_wind_last;

    Real e_wind = L_wind / (4 * PI * rin * rin * v_wind);

    Real B2_wind = e_wind / (1 + 1 / sigma_wind - 1.0 / 2.0 / gamma_wind / gamma_wind -
                             hydro_coef * k_wind / (hydro_coef * k_wind + 1) / gamma_wind / gamma_wind / sigma_wind);

    rho_wind = B2_wind / gamma_wind / gamma_wind / sigma_wind / (hydro_coef * k_wind + 1);

    p_wind = k_wind * rho_wind;

    B_wind = sqrt(B2_wind / 2);
    // ejecta calculations

    rho_ej = M_ej / (r_c * r_c * r_c * 2 * PI * (0.5 + 3.0 / 8 * PI));

    rho_tail = 0.01 * M_ej / (4 * PI * r_c * r_c * r_c * (pow(4, 3 - tail_n) - 1) / (3 - tail_n));

    Real E_ej = 0.5 * rho_ej * v_ej * v_ej * 2 * PI * (0.5 + 3.0 / 8 * PI) * r_c * r_c * r_c / 3;

    // jet calculations

    Real gamma_jet_r = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r);  // gamma in radial direction

    Real w = Gamma_inf / gamma_jet_r;

    Real eta = w / (sigma_jet + 1);

    Real e_jet = L_jet / (v_jet_r * rin * rin * 4 * PI);  // isotropic energy

    rho_jet = e_jet / (gamma_jet_r * gamma_jet_r * w - 0.5 * w / (1 + 1 / sigma_jet) - (eta - 1) / hydro_coef);

    p_jet = (eta - 1) / hydro_coef * rho_jet;

    Real b2 = rho_jet * w / (1 / sigma_jet + 1);

    B_jm = gamma_jet_r * sqrt(b2);

    /*Real b2 = (B_jm * B_jm) / gamma_jet_r / gamma_jet_r;

    Real pm_jet = 0.5 * b2;

    p_jet = (e_jet + pm_jet - gamma_jet_r * gamma_jet_r * b2 * w / (w - 1)) /
            (hydro_coef * w / (w - 1) * gamma_jet_r * gamma_jet_r - 1);

    rho_jet = (hydro_coef * p_jet + b2) / (w - 1);

    Real eta = w - b2 / rho_jet;

    Real sigma = w / eta - 1;*/

    const Real uL = 3e10;
    const Real uM = 1.989e33;
    const Real uT = 1;
    const Real uP = 6.66666e22;

    const Real urho = 74;

    const Real uB = 9.15e11;
    const Real uE = 1.8e54;

    // print_par("p_amb", p_amb);

    print_par("E_ej", E_ej * uE, E_ej);
    print_par("rho_ej", rho_ej * urho, rho_ej);
    print_par("rho_tail", rho_tail * urho, rho_tail);

    print_par("L_jet", L_jet * uE / uT, L_jet);
    print_par("rho_jet", rho_jet * urho, rho_jet);
    print_par("sigma_jet", sigma_jet, sigma_jet);
    print_par("B_jet", B_jm * uB, B_jm);
    print_par("p_jet", p_jet * uP, p_jet);

    print_par("w", 1 + hydro_coef * p_jet / rho_jet + b2 / rho_jet, w);
    print_par("eta", 1 + hydro_coef * p_jet / rho_jet, eta);

    print_par("rho_wind", rho_wind * urho, rho_wind);
    print_par("B_wind", B_wind * uB, B_wind);
    print_par("L_wind", L_wind * uE / uT, L_wind);
    print_par("E_wind", E_wind * uE, E_wind);

    if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(BoundaryFace::inner_x1, LoopInnerX1);
    }
    return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
            for (int i = is; i <= ie; i++) {
                if (ej_on) {
                    Real r = pcoord->x1f(i);
                    Real sin_theta = std::sin(pcoord->x2v(j));
                    Real v = 0;
                    Real rho = 0;
                    Real p = 0;
                    if (r < r_c) {
                        rho = rho_ej * (0.25 + sin_theta * sin_theta * sin_theta) * pow(r / r_c, -2);
                        v = v_ej * r / r_c;
                        p = k_ej * pow(rho, gamma_hydro);
                    } else if (r < 4 * r_c) {
                        rho = rho_tail * pow(r / r_c, -tail_n);
                        v = v_ej;
                        p = k_ej * pow(rho, gamma_hydro);
                    } else {
                        rho = rho_amb;
                        v = 0;
                        p = p_amb;
                    }
                    Real g = 1.0 / sqrt(1 - v * v);
                    phydro->u(IDN, k, j, i) = g * rho;
                    phydro->u(IM1, k, j, i) = g * g * (rho + hydro_coef * p) * v;
                    phydro->u(IM2, k, j, i) = 0.0;
                    phydro->u(IM3, k, j, i) = 0.0;
                    phydro->u(IEN, k, j, i) = g * g * (rho + hydro_coef * p) - p;
                } else {
                    phydro->u(IDN, k, j, i) = rho_amb;
                    phydro->u(IM1, k, j, i) = 0.0;
                    phydro->u(IM2, k, j, i) = 0.0;
                    phydro->u(IM3, k, j, i) = 0.0;
                    phydro->u(IEN, k, j, i) = (rho_amb + hydro_coef * p_amb) - p_amb;
                }
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

inline Real transit(Real t, Real t0, Real width) { return 1 / (1 + std::exp((t - t0) / width)); }

void LoopInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &b, Real time, Real dt, int il,
                 int iu, int jl, int ju, int kl, int ku, int ngh) {
    if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") != 0) {
        std::cout << "use sphereical coordinate" << std::endl;
        exit(0);
    }

    if (jet_on && time < t_jet_duration) {
        // std::cout << "jet launch t= " << time << "\n";
        if (MAGNETIC_FIELDS_ENABLED) {
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
                        if (pcoord->x2v(j) < theta_jet) {
                            b.x3f(k, j, (il - i)) = B_jm;
                        } else {
                            b.x3f(k, j, (il - i)) = 0.0;
                        }
                    }
                }
            }
        }

        for (int k = kl; k <= ku; ++k) {
            for (int j = ju; j >= jl; --j) {
                for (int i = 1; i <= ngh; ++i) {
                    if (pcoord->x2v(j) < theta_jet) {
                        Real gamma_jet = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r);
                        prim(IDN, k, j, il - i) = rho_jet;
                        prim(IVX, k, j, il - i) = gamma_jet * v_jet_r;
                        prim(IVY, k, j, il - i) = 0.0;
                        prim(IVZ, k, j, il - i) = 0;
                        prim(IPR, k, j, il - i) = p_jet;
                    } else {
                        prim(IDN, k, j, il - i) = prim(IDN, k, j, il + i - 1);
                        prim(IVX, k, j, il - i) = -prim(IVX, k, j, il + i - 1);
                        prim(IVY, k, j, il - i) = prim(IVY, k, j, il + i - 1);
                        prim(IVZ, k, j, il - i) = prim(IVZ, k, j, il + i - 1);
                        prim(IPR, k, j, il - i) = prim(IPR, k, j, il + i - 1);
                    }
                }
            }
        }
    } else if (time < t_wind_launch) {
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    prim(IDN, k, j, il - i) = prim(IDN, k, j, il + i - 1);
                    prim(IVX, k, j, il - i) = -prim(IVX, k, j, il + i - 1);
                    prim(IVY, k, j, il - i) = prim(IVY, k, j, il + i - 1);
                    prim(IVZ, k, j, il - i) = prim(IVZ, k, j, il + i - 1);
                    prim(IPR, k, j, il - i) = prim(IPR, k, j, il + i - 1);
                }
            }
        }
        if (MAGNETIC_FIELDS_ENABLED) {
            SET_MAGNETIC_FIELD_BC_REFLECTING
        }
    } else if ((time >= t_wind_launch) && (time <= (t_wind_launch + t_wind_last))) {
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    prim(IDN, k, j, il - i) = rho_wind;
                    prim(IVX, k, j, il - i) = v_wind / sqrt(1 - v_wind * v_wind);
                    prim(IVY, k, j, il - i) = 0.0;
                    prim(IVZ, k, j, il - i) = 0.0;
                    prim(IPR, k, j, il - i) = p_wind;
                }
            }
        }

        if (MAGNETIC_FIELDS_ENABLED) {
            static std::atomic_bool radial_b = ATOMIC_VAR_INIT(false);

            if (radial_b == false) {
                for (int k = kl; k <= ku; ++k) {
                    for (int j = jl; j <= ju; ++j) {
                        for (int i = il - ngh; i <= iu + ngh; ++i) {
                            b.x1f(k, j, i) += B_wind * rin * rin / pcoord->x1f(i) / pcoord->x1f(i);
                        }
                    }
                }
                radial_b = true;
            }

            for (int k = kl; k <= ku; ++k) {
                for (int j = jl; j <= ju; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        b.x1f(k, j, (il - i)) = -b.x1f(k, j, il + i - 1);
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
                        Real theta = pcoord->x2v(j);
                        Real sinx = sin(theta);
                        b.x3f(k, j, (il - i)) = -B_wind * sinx * rin / pcoord->x1f(il - i) *
                                                1.2247;  // 1.2247 = sqrt(3/2) //B_wind is average value
                    }
                }
            }
        }
    } else if (time > (t_wind_launch + t_wind_last)) {
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    prim(IDN, k, j, il - i) = prim(IDN, k, j, il + i - 1);
                    prim(IVX, k, j, il - i) = -prim(IVX, k, j, il + i - 1);
                    prim(IVY, k, j, il - i) = prim(IVY, k, j, il + i - 1);
                    prim(IVZ, k, j, il - i) = prim(IVZ, k, j, il + i - 1);
                    prim(IPR, k, j, il - i) = prim(IPR, k, j, il + i - 1);
                }
            }
        }
        if (MAGNETIC_FIELDS_ENABLED) {
            SET_MAGNETIC_FIELD_BC_OUTFLOW
        }
    }
}
