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

Real rho_amb = 0;
// Real p_amb = 0;
Real rin = 1e-3;

Real rho_ej = 0;
// Real p_ej = 0;
Real v_ej = 0;
Real t_ej_crit = 0.0;
Real t_ej_end = 0;
Real gamma_ej = 1.0;

Real theta_jet = 0;
Real t_jet_launch = 0;
Real t_jet_duration = 0;

Real rho_jet = 0;
Real p_jet_ave = 0;
Real v_jet_r = 0.0;
Real v_jet_m = 0.0;
Real gamma_jet_r = 1.0;
Real B_r = 0.0;
Real B_phi = 0.0;
const Real Alpha = 0.2;

Real t_wind_launch = 0.5;
Real rho_wind = 0;
Real Omega = 0.2;

Real wind_dt = 0;
Real B_wind = 0;

Real Gamma = 1.333333333333333333;

Real K_EFF = 4.76E-7;

size_t data_size = 1000;
std::vector<Real> THETA;
std::vector<Real> P;

Real sigmod(Real x, Real x0, Real width) { return 1 / (1 + std::exp((x - x0) / width)); }

Real ave_coef_b2_phi(Real alpha) {
    return 4 * alpha * alpha * (log(1 + 1 / (alpha * alpha)) - 1 / (1 + alpha * alpha));
}

Real calc_p_b(Real Br, Real Bphi, Real vr, Real vphi) {
    Real gr = 1.0 / std::sqrt(1 - vr * vr);
    return Br * Br / 2 * (1 - vphi * vphi) + Bphi * Bphi / 2 / gr / gr + vr * Br * vphi * Bphi;
}

inline Real get_Bphi(Real theta, Real Bjm, Real theta_j, Real alpha) {
    Real theta_m = theta_j * alpha;
    Real ratio = theta / theta_m;
    return 2 * Bjm * ratio / (1 + ratio * ratio);
}

inline Real dBdtheta(Real theta, Real Bjm, Real theta_j, Real alpha) {
    Real theta_m = theta_j * alpha;
    Real ratio = theta / theta_m;
    return 2 * Bjm * (1 - ratio * ratio) / (1 + ratio * ratio) / (1 + ratio * ratio) / theta_m;
}

inline Real get_vphi(Real theta, Real vjm, Real theta_j) {
    Real ratio = theta / theta_j;
    return vjm * ratio;
}

inline Real dvdtheta(Real theta, Real vjm, Real theta_j) { return vjm / theta_j; }

Real dpdtheta(Real theta, Real rho, Real p_now, Real Br, Real Bjm, Real vr, Real vjm, Real alpha, Real theta_j) {
    Real Bphi = get_Bphi(theta, Bjm, theta_j, alpha);
    Real vphi = get_vphi(theta, vjm, theta_j);
    Real dBdtheta_ = dBdtheta(theta, Bjm, theta_j, alpha);
    Real dvdtheta_ = dvdtheta(theta, vjm, theta_j);
    Real gr = 1.0 / std::sqrt(1 - vr * vr);
    Real g = 1.0 / std::sqrt(1 - vr * vr - vphi * vphi);
    Real h = 1 + 4 * p_now / rho;

    return -Bphi * Bphi / theta / gr / gr - (Bphi / gr / gr + Br * vr * vphi) * dBdtheta_ +
           (Br * vphi - vr * Bphi) * Br * dvdtheta_ + rho * h * g * g * vphi * vphi / theta +
           Br * vphi / theta * (Br * vphi - 2 * Bphi * vr);
}

void calc_p_jet_profile(Real rho, Real Br, Real Bjm, Real vr, Real vjm, Real pa, Real alpha, Real theta_j) {
    Real Bphi1 = get_Bphi(theta_j, Bjm, theta_j, alpha);
    Real vphi1 = get_vphi(theta_j, vjm, theta_j);
    Real p1 = pa - calc_p_b(Br, Bphi1, vr, vphi1);

    THETA.resize(data_size + 1);
    P.resize(data_size + 1);
    size_t len = THETA.size() - 1;

    std::cout << "p1 = " << p1 << "pa =" << pa << "rho = " << rho << std::endl;

    for (size_t i = 0; i <= len; i++) {
        THETA[i] = theta_j * static_cast<Real>(i) / static_cast<Real>(len);
        P[i] = p1;
        // std::cout << "theta = " << THETA[i] << ", p = " << P[i] << std::endl;
    }
    Real dtheta = THETA[1] - THETA[0];
    for (int i = data_size - 1; i >= 0; i--) {
        P[i] = P[i + 1] - dpdtheta(THETA[i + 1], rho, P[i + 1], Br, Bjm, vr, vjm, alpha, theta_j) * dtheta;
        if (P[i] < 0) P[i] = 0;
    }
    for (size_t i = 0; i < P.size(); i += 10) {
        std::cout << "theta = " << THETA[i] << ", p = " << P[i] << std::endl;
    }
}

Real get_p_value(std::vector<Real> &theta, std::vector<Real> &p, Real theta_now) {
    if (theta_now < theta[0]) {
        return p[0];
    }
    if (theta_now > theta.back()) {
        return p.back();
    }
    size_t i = 0;
    while (theta[i] < theta_now) {
        i++;
    }
    Real p1 = p[i - 1];
    Real p2 = p[i];
    Real theta1 = theta[i - 1];
    Real theta2 = theta[i];
    return p1 + (p2 - p1) * (theta_now - theta1) / (theta2 - theta1);
}

Real calc_p_ave(std::vector<Real> &theta, std::vector<Real> &p) {
    Real p_ave = 0;
    for (size_t i = 0; i < theta.size() - 1; i++) {
        p_ave += (p[i] + p[i + 1]) / 2 * ((theta[i + 1] + theta[i]) / 2) * (theta[i + 1] - theta[i]);
    }
    Real theta_max = theta[theta.size() - 1];
    return p_ave / (theta_max * theta_max / 2);
}

bool check_p(std::vector<Real> &p) {
    for (auto pi : p) {
        if (pi < 0) {
            return false;
        }
    }
    return true;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
    K_EFF = pin->GetOrAddReal("problem", "k_eff", 1e-6);
    rho_amb = pin->GetOrAddReal("problem", "rho_amb", 1e-10);
    // p_amb = pin->GetOrAddReal("problem", "p_amb", 1e-10);
    rin = pin->GetOrAddReal("problem", "r_in", 1.6e-3);

    Real M_ej = pin->GetOrAddReal("problem", "M_ej", 0.01);
    v_ej = pin->GetOrAddReal("problem", "v_ej", 0.2);
    // p_ej = pin->GetOrAddReal("problem", "p_ej", 16);
    gamma_ej = 1 / std::sqrt(1 - v_ej * v_ej);

    t_ej_end = pin->GetOrAddReal("problem", "t_ej_duration", 0.015);
    t_ej_crit = pin->GetOrAddReal("problem", "t_ej_crit", 0.005);

    Real int_coef = 2 * PI * (0.5 + 3.0 / 8 * PI) * t_ej_crit;  // (2 * t_ej_crit - t_ej_crit * t_ej_crit / t_ej_end);
    rho_ej = M_ej / (v_ej * rin * rin * int_coef);

    theta_jet = pin->GetOrAddReal("problem", "theta_jet", 0.17453292519943295);
    t_jet_launch = pin->GetOrAddReal("problem", "t_jet_launch", 0.5);
    t_jet_duration = pin->GetOrAddReal("problem", "t_jet_duration", 1);
    v_jet_r = pin->GetOrAddReal("problem", "v_jet_r", 0.8);
    v_jet_m = pin->GetOrAddReal("problem", "v_jet_phi", 0.4);

    gamma_jet_r = 1 / std::sqrt(1 - v_jet_r * v_jet_r);

    Real Gamma_inf = pin->GetOrAddReal("problem", "Gamma_inf", 265);
    Real L_jet = pin->GetOrAddReal("problem", "L_jet", 0.00278);

    Real e_jet = L_jet / (v_jet_r * rin * rin * 4 * PI);

    B_r = pin->GetOrAddReal("problem", "B_r", 11.6);
    B_phi = pin->GetOrAddReal("problem", "B_phi", 23.2);
    // Real b2 = B_r * B_r + B_phi * B_phi * ave_coef_b2_phi(Alpha) / gamma_jet_r / gamma_jet_r;
    Real b2_r = (B_r * B_r + ave_coef_b2_phi(Alpha) * B_phi * B_phi) / gamma_jet_r / gamma_jet_r +
                (v_jet_r * B_r + ave_coef_b2_phi(Alpha) * B_phi * v_jet_m) *
                    (v_jet_r * B_r + ave_coef_b2_phi(Alpha) * B_phi * v_jet_m);
    Real pm = b2_r / 2;
    Real h_star = Gamma_inf / gamma_jet_r;

    rho_jet = e_jet / gamma_jet_r / gamma_jet_r / h_star;

    Real h = h_star - b2 / rho_jet / 2;

    p_jet_ave = (h - 1) * rho_jet / 4;

    Real sigma_r = B_r * B_r / (p_jet_ave) / 2;
    Real sigma_phi = B_phi * B_phi / 2 / (p_jet_ave)*ave_coef_b2_phi(Alpha) / gamma_jet_r / gamma_jet_r;

    t_wind_launch = pin->GetOrAddReal("problem", "t_wind_launch", 0.5);
    Omega = pin->GetOrAddReal("problem", "Omega", 0.2);
    wind_dt = PI / (Omega);
    B_wind = pin->GetOrAddReal("problem", "B_mag", 387) * 1e6 / 3e10 / rin;
    rho_wind = pin->GetOrAddReal("problem", "rho_wind", 1);

    std::cout << rho_ej << ' ' << rho_jet << ' ' << sigma_r << " " << sigma_phi << ' ' << h_star << ' ' << p_jet_ave
              << std::endl;

    if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(BoundaryFace::inner_x1, LoopInnerX1);
    }

    return;
}

Real calc_jet_rho(Real e, Real Gamma, Real b2, Real vB, Real sigma){
    
}
//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Spherical blast wave test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    Gamma = pin->GetOrAddReal("hydro", "gamma", 1.33333333);
    Real Gamma_1 = Gamma - 1.0;
    Real r_init = rin + v_ej * t_ej_crit;
    Real p_amb = K_EFF * pow(rho_amb, Gamma);
    for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
            for (int i = is; i <= ie; i++) {
                phydro->u(IDN, k, j, i) = rho_amb;
                phydro->u(IM1, k, j, i) = 0.0;
                phydro->u(IM2, k, j, i) = 0.0;
                phydro->u(IM3, k, j, i) = 0.0;
                phydro->u(IEN, k, j, i) = (rho_amb + Gamma / Gamma_1 * p_amb) - p_amb;
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
    Real p_a = K_EFF * rho_ej * (0.25 + sin(theta_jet) * sin(theta_jet) * sin(theta_jet));
    calc_p_jet_profile(rho_jet, B_r, B_phi, v_jet_r, v_jet_m, p_a, Alpha, theta_jet);

    bool p_positive = check_p(P);
    if (p_positive == false) {
        std::cout << "p is negative" << std::endl;
        exit(0);
    }

    Real p_ave_real = calc_p_ave(THETA, P);
    Real hh = 1 + 4 * p_ave_real / rho_jet;
    Real sigma_r = B_r * B_r / 2 / p_ave_real;
    Real sigma_phi = ave_coef_b2_phi(Alpha) * B_phi * B_phi / 2 / p_ave_real / gamma_jet_r / gamma_jet_r;
    std::cout << "launching jet at t = " << time << std::endl;
    std::cout << "p_ave = " << p_ave_real << std::endl;
    std::cout << "h = " << hh << std::endl;
    std::cout << "sigma_r = " << sigma_r << std::endl;
    std::cout << "sigma_phi = " << sigma_phi << std::endl;
    std::cout << "h* = " << hh + 0.5 * (hh - 1) * (sigma_phi + sigma_r) << std::endl;
}

size_t get_boundary_index(Coordinates *pcoord, size_t jl, size_t ju, Real theta_jet) {
    for (int j = jl; j <= ju; ++j) {
        if (pcoord->x2v(j) > theta_jet) {
            return j;
        }
    }
    return ju;
}

void LoopInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &b, Real time, Real dt, int il,
                 int iu, int jl, int ju, int kl, int ku, int ngh) {
    if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") != 0) {
        std::cout << "use sphereical coordinate" << std::endl;
        exit(0);
    }
    /*const Real rr = (rin / 0.04333);
    const Real tau = rr * t_ej_crit / (1 - rr);
     if (time < t_ej_crit) {
         // Real vel = v_ej * tau / (time + tau);
         // Real gamma_ej_t = 1 / std::sqrt(1 - vel * vel);
         Real vel = v_ej;
         //*tau / (time + tau);
         Real gamma_ej_t = 1 / std::sqrt(1 - vel * vel);
         for (int k = kl; k <= ku; ++k) {
             for (int j = jl; j <= ju; ++j) {
                 for (int i = 1; i <= ngh; ++i) {
                     Real sin_theta = std::sin(pcoord->x2v(j));
                     prim(IDN, k, j, il - i) = rho_ej * (0.25 + sin_theta * sin_theta * sin_theta);
                     prim(IVX, k, j, il - i) = gamma_ej_t * vel;
                     prim(IVY, k, j, il - i) = 0.0;
                     prim(IVZ, k, j, il - i) = 0.0;
                     prim(IPR, k, j, il - i) = K_EFF * pow(prim(IDN, k, j, il - i), Gamma);
                 }
             }
         }

         if (MAGNETIC_FIELDS_ENABLED) {
             SET_MAGNETIC_FIELD_BC_OUTFLOW
         }
     } else if (time <= t_ej_end) {
         Real vel = v_ej * tau / (t_ej_crit + tau);
         Real gamma_ej_t = 1 / std::sqrt(1 - vel * vel);
         for (int k = kl; k <= ku; ++k) {
             for (int j = jl; j <= ju; ++j) {
                 for (int i = 1; i <= ngh; ++i) {
                     Real sin_theta = std::sin(pcoord->x2v(j));
                     prim(IDN, k, j, il - i) =
                         rho_ej * (0.25 + sin_theta * sin_theta * sin_theta) * t_ej_crit * t_ej_crit / time / time;
                     prim(IVX, k, j, il - i) = gamma_ej_t * vel;
                     // prim(IVX, k, j, il - i) = gamma_ej * v_ej;
                     prim(IVY, k, j, il - i) = 0.0;
                     prim(IVZ, k, j, il - i) = 0.0;
                     prim(IPR, k, j, il - i) = K_EFF * pow(prim(IDN, k, j, il - i), Gamma);
                 }
             }
         }
         if (MAGNETIC_FIELDS_ENABLED) {
             SET_MAGNETIC_FIELD_BC_OUTFLOW
         }
     }*/
    const Real rr = (rin / 0.04333);
    const Real tau = rr * t_ej_end / (1 - rr);
    if (time < t_ej_end) {
        Real vel = v_ej * tau / (time + tau);
        Real gamma_ej_t = 1 / std::sqrt(1 - vel * vel);
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    Real sin_theta = std::sin(pcoord->x2v(j));
                    prim(IDN, k, j, il - i) = rho_ej * (0.25 + sin_theta * sin_theta * sin_theta);
                    prim(IVX, k, j, il - i) = gamma_ej_t * vel;
                    prim(IVY, k, j, il - i) = 0.0;
                    prim(IVZ, k, j, il - i) = 0.0;
                    prim(IPR, k, j, il - i) = K_EFF * pow(prim(IDN, k, j, il - i), Gamma);
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

    if ((time >= t_jet_launch + t_ej_end) && (time < t_ej_end + t_jet_launch + t_jet_duration)) {
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
                            b.x2f(k, j, (il - i)) = b.x1f(k, j, (il));
                        }
                    }
                }
            }
            for (int k = kl; k <= ku + 1; ++k) {
                for (int j = jl; j <= ju; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        if (pcoord->x2v(j) < theta_jet) {
                            b.x3f(k, j, (il - i)) = get_Bphi(pcoord->x2v(j), B_phi, theta_jet, Alpha);
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
                    if (pcoord->x2v(j) < theta_jet) {
                        /* Real theta_r = pcoord->x2v(j) / theta_jet;
                         Real theta_r2 = theta_r * theta_r;
                         Real theta_r3 = theta_r2 * theta_r;
                         Real theta_r4 = theta_r3 * theta_r;
                         Real theta_r5 = theta_r4 * theta_r;*/
                        Real v_phi = get_vphi(pcoord->x2v(j), v_jet_m, theta_jet);
                        Real gamma_jet = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r - v_phi * v_phi);

                        prim(IDN, k, j, il - i) = rho_jet;
                        prim(IVX, k, j, il - i) = gamma_jet * v_jet_r;
                        prim(IVY, k, j, il - i) = 0.0;
                        prim(IVZ, k, j, il - i) = gamma_jet * v_phi;

                        // Real p_index = 11.5961 * theta_r5 - 30.9 * theta_r4 + 30.6 * theta_r3 - 10.4 * theta_r2 -
                        //                1.285 * theta_r + 6.539;
                        prim(IPR, k, j, il - i) =
                            get_p_value(THETA, P, pcoord->x2v(j));  // p_jet_ave / (615389) * pow(10.0, p_index);
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
    } else if (time >= t_ej_end + t_jet_launch + t_jet_duration) {
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

    if (time >= t_wind_launch) {
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
    }

    // copy face-centered magnetic fields into ghost zones
}
