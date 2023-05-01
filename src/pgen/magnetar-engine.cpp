//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
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
Real r_c = 0.04333333;
bool ej_on = true;

// jet
Real theta_jet = 0.1;
Real t_jet_duration = 1;
Real v_jet_r = 0.8;
Real rho_jet = 0;
Real p_jet = 0;
Real B_jm = 0;
bool jet_on = true;
// wind
Real t_wind_launch = 0;
Real t_wind_last = 10;
Real B_star = 0;
Real r_star = 1.2e6 / 3e10;
Real Omega = 100;
Real monopole_dist = 0;
Real sigma_wind = 100;
Real gamma_wind = 100;
Real rho_wind = 0;
Real k_wind = 1e-5;

inline void print_par(std::string name, Real value, Real code_val) {
    std::cout << name << " = " << value << ',' << code_val << std::endl;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
    rin = pin->GetOrAddReal("problem", "r_in", 1.6666666666e-3);
    // rin = pcoord->x1v(is);
    //  reading hydro paramters
    gamma_hydro = pin->GetOrAddReal("hydro", "gamma", 1.33333333);
    hydro_coef = gamma_hydro / (gamma_hydro - 1);

    // reading parameters of ambient medium
    p_amb = pin->GetOrAddReal("problem", "p_amb", 1e-15);
    rho_amb = pin->GetOrAddReal("problem", "rho_amb", 1.35e-6);

    // reading parameters of ejecta
    ej_on = pin->GetBoolean("problem", "ej_on");
    Real M_ej = pin->GetOrAddReal("problem", "M_ej", 0.01);
    v_ej = pin->GetOrAddReal("problem", "v_ej", 0.2);
    Real t_d = pin->GetOrAddReal("problem", "t_d", 0.5);

    // reading parameters of jet
    jet_on = pin->GetBoolean("problem", "jet_on");
    theta_jet = pin->GetOrAddReal("problem", "theta_jet", 0.17453292519943295);
    t_jet_duration = pin->GetOrAddReal("problem", "t_jet_duration", 1);
    v_jet_r = pin->GetOrAddReal("problem", "v_jet_r", 0.8);
    Real Gamma_inf = pin->GetOrAddReal("problem", "Gamma_inf", 300);
    Real L_jet = pin->GetOrAddReal("problem", "L_jet", 0.00278);
    Real sigma_jet = pin->GetOrAddReal("problem", "sigma_jet", 10);

    // reading parameters of wind
    t_wind_launch = pin->GetOrAddReal("problem", "t_wind_launch", 6);
    t_wind_last = pin->GetOrAddReal("problem", "t_wind_last", 10);
    // Real E_wind = pin->GetOrAddReal("problem", "E_wind", 6.66666e-6);
    B_star = pin->GetOrAddReal("problem", "B_star", 100);
    sigma_wind = pin->GetOrAddReal("problem", "sigma_wind", 1000);
    Real T = pin->GetOrAddReal("problem", "T", 0.001);
    k_wind = pin->GetOrAddReal("problem", "k_wind", 0.05);
    monopole_dist = pin->GetOrAddReal("problem", "monopole_dist", 0);

    /// initializing variables
    Omega = 2 * PI / T;

    Real Rlc = 1 / Omega;

    // Real B_in = B_star * r_star * r_star / rin / rin;

    Real B_wind_phi = B_star * r_star * r_star / rin / Rlc;

    Real cs2 = gamma_hydro * k_wind;

    Real gamma_wind = sqrt((1 + sigma_wind) / (1 - cs2));

    rho_wind = B_wind_phi * B_wind_phi / gamma_wind / gamma_wind / sigma_wind;

    Real L_wind = 2 * B_star * B_star * r_star * r_star * r_star * r_star * Omega * Omega / 3;
    // ejecta calculations

    r_c = t_d * v_ej;

    // Real v_ej_esc = 0.024 * pow(t_d, -1.0 / 3);  // 0.024c is the escape velocity of t_d = 1sec with M_ns = 1.4Msun

    // r_ej_in = v_ej_esc * t_d;

    rho_ej = M_ej / (r_c * r_c * 2 * PI * (0.5 + 3 * PI / 8) * (r_c - rin));
    // Real E_ej = 0.5 * rho_ej * v_ej * v_ej * 4 * PI * r_c * r_c * r_c / 3;
    // jet calculations

    Real gamma_jet_r = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r);  // gamma in radial direction

    Real w = Gamma_inf / gamma_jet_r;

    Real eta = w / (sigma_jet + 1);

    if (sigma_jet > 1) {
        eta = w * (1 + pow(sigma_jet, 2.0 / 3.0)) / (sigma_jet + 1);
    }

    Real e_jet = L_jet / (v_jet_r * rin * rin * 4 * PI);  // isotropic energy

    rho_jet = e_jet / (gamma_jet_r * gamma_jet_r * w - 0.5 * w / (1 + 1 / sigma_jet) - (eta - 1) / hydro_coef);

    p_jet = (eta - 1) / hydro_coef * rho_jet;

    Real b2 = rho_jet * w / (1 / sigma_jet + 1);

    B_jm = gamma_jet_r * sqrt(b2);

    const Real uL = 3e10;
    const Real uM = 1.989e33;
    const Real uT = 1;
    const Real uP = 6.66666e22;
    const Real urho = 74;
    const Real uB = 9.15e11;
    const Real uE = 1.8e54;
    // print_par("p_amb", p_amb);
    // print_par("E_ej", E_ej * uE, E_ej);

    print_par("rho_ej", rho_ej * urho, rho_ej);
    print_par("rho_ej_inner", rho_ej * r_c * r_c / rin / rin * urho, rho_ej * r_c * r_c / rin / rin);
    print_par("rho_jet", rho_jet * urho, rho_jet);
    print_par("gamma_wind", gamma_wind, gamma_wind);
    print_par("rho_wind", rho_wind * urho, rho_wind);
    print_par("p_wind", k_wind * rho_wind * uP, rho_wind * k_wind);

    print_par("sigma_jet", sigma_jet, sigma_jet);
    print_par("sigma_wind", sigma_wind, sigma_wind);

    print_par("L_jet", L_jet * uE / uT, L_jet);
    print_par("L_wind", L_wind * uE / uT, L_wind);

    print_par("B_jet", B_jm * uB, B_jm);
    print_par("B_in_wind", B_wind_phi * uB, B_wind_phi);
    print_par("B_star", B_star * uB, B_star);

    print_par("w", 1 + hydro_coef * p_jet / rho_jet + b2 / rho_jet, w);
    print_par("eta", 1 + hydro_coef * p_jet / rho_jet, eta);
    print_par("E_wind", L_wind * t_wind_last * uE, L_wind * t_wind_last);

    if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(BoundaryFace::inner_x1, LoopInnerX1);
    }
    return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    //----------------------------------------------------------------------------------------------
    for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
            for (int i = is; i <= ie; i++) {
                if (ej_on) {
                    Real r = pcoord->x1v(i);
                    Real sin_theta = std::sin(pcoord->x2v(j));
                    Real v = 0;
                    Real rho = 0;
                    Real p = p_amb;
                    if (r < r_c) {
                        rho = rho_ej * (0.25 + sin_theta * sin_theta * sin_theta) * pow(r / r_c, -2);
                        v = v_ej * r / r_c;
                    } else {
                        rho = rho_amb;
                        v = 0;
                    }
                    Real g = 1.0 / sqrt(1 - v * v);
                    phydro->u(IDN, k, j, i) = g * rho;
                    phydro->u(IM1, k, j, i) = g * g * (rho + hydro_coef * p) * v;
                    phydro->u(IM2, k, j, i) = 0.0;
                    phydro->u(IM3, k, j, i) = 0.0;
                    phydro->u(IEN, k, j, i) = g * g * (rho + hydro_coef * p) - p;
                } else {
                    Real r = pcoord->x1v(i);
                    Real Br = 0;
                    //          B_star *r_star *r_star / r / r;
                    phydro->u(IDN, k, j, i) = rho_amb;
                    phydro->u(IM1, k, j, i) = 0.0;
                    phydro->u(IM2, k, j, i) = 0.0;
                    phydro->u(IM3, k, j, i) = 0.0;
                    phydro->u(IEN, k, j, i) = (rho_amb + hydro_coef * p_amb) - p_amb + Br * Br * 0.5;
                }
            }
        }
    }

    for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
            for (int i = is; i <= ie + 1; ++i) {
                pfield->b.x1f(k, j, i) = 0.0;
                // B_star *r_star *r_star / pcoord->x1f(i) / pcoord->x1f(i);
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

void LoopInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &b, Real time, Real dt, int il,
                 int iu, int jl, int ju, int kl, int ku, int ngh) {
    if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") != 0) {
        std::cout << "use sphereical coordinate" << std::endl;
        exit(0);
    }
    Real Rlc = 1 / Omega;
    if (time < t_jet_duration) {
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
                    if (jet_on && (pcoord->x2v(j) < theta_jet || pcoord->x2v(j) > (PI - theta_jet))) {
                        b.x3f(k, j, (il - i)) = -B_jm;
                    } else {
                        b.x3f(k, j, (il - i)) = 0.0;
                    }
                }
            }
        }

        for (int k = kl; k <= ku; ++k) {
            for (int j = ju; j >= jl; --j) {
                for (int i = 1; i <= ngh; ++i) {
                    Real theta = pcoord->x2v(j);
                    if (jet_on && (theta < theta_jet || theta > (PI - theta_jet))) {
                        Real gamma_jet = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r);
                        prim(IDN, k, j, il - i) = rho_jet;
                        prim(IVX, k, j, il - i) = gamma_jet * v_jet_r;
                        prim(IVY, k, j, il - i) = 0.0;
                        prim(IVZ, k, j, il - i) = 0.0;
                        prim(IPR, k, j, il - i) = p_jet;
                    } else {
                        Real sin_theta = sin(theta);
                        Real r = pcoord->x1v(il - i);
                        Real rho = rho_ej * (0.25 + sin_theta * sin_theta * sin_theta) * pow(r / r_c, -2);
                        Real v_r = v_ej * r / r_c;
                        prim(IDN, k, j, il - i) = rho;
                        prim(IVX, k, j, il - i) = v_r / sqrt(1 - v_r * v_r);
                        prim(IVY, k, j, il - i) = 0;
                        prim(IVZ, k, j, il - i) = 0;
                        prim(IPR, k, j, il - i) = p_amb;
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
            SET_MAGNETIC_FIELD_BC_REFLECTING  // avoid continuous poynting flux
        }
    } else if ((time >= t_wind_launch) && (time <= (t_wind_launch + t_wind_last))) {
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    Real sin_theta = std::sin(pcoord->x2v(j));
                    Real r = pcoord->x1v(il - i);
                    Real Bphi = -B_star * r_star * r_star / r / Rlc * sin_theta;
                    Real cs2 = gamma_hydro * k_wind;
                    Real A = Bphi * Bphi * (1 - cs2) / rho_wind;
                    Real sigma = 0.5 * (sqrt(1 + 4 * A) - 1);
                    Real g = sqrt((1 + sigma) / (1 - cs2));
                    Real ur = sqrt(g * g - 1);
                    prim(IDN, k, j, il - i) = rho_wind;
                    prim(IVX, k, j, il - i) = ur;
                    prim(IVY, k, j, il - i) = 0.0;
                    prim(IVZ, k, j, il - i) = 0.0;
                    prim(IPR, k, j, il - i) = rho_wind * k_wind;
                    /* Real g = sqrt((1 + sigma_wind) / (1 - cs2));
                     Real ur = sqrt(g * g - 1);
                     Real rho = Bphi * Bphi / sigma_wind / g / g;
                     prim(IDN, k, j, il - i) = rho;
                     prim(IVX, k, j, il - i) = ur;
                     prim(IVY, k, j, il - i) = 0.0;
                     prim(IVZ, k, j, il - i) = 0.0;
                     prim(IPR, k, j, il - i) = rho * k_wind;*/
                }
            }
        }

        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    b.x1f(k, j, (il - i)) = 0.0;
                    // b.x1f(k, j, (il - i)) = B_star * r_star * r_star / pcoord->x1v(il - i) / pcoord->x1v(il - i);
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
                    Real r = pcoord->x1v(il - i);
                    Real theta = pcoord->x2f(j);
                    b.x3f(k, j, (il - i)) = -B_star * r_star * r_star / r / Rlc * sin(theta);
                }
            }
        }
    }
}
