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
void Jet(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b, Real time, Real dt, int il, int iu,
         int jl, int ju, int kl, int ku, int ngh);

const Real Alpha = 0.2;

size_t data_size = 1000;
std::vector<Real> THETA;
std::vector<Real> P;

Real sigmod(Real x, Real x0, Real width) { return 1 / (1 + std::exp((x - x0) / width)); }

Real ave_coef_b2_phi(Real alpha) {
    return 4 * alpha * alpha * (log(1 + 1 / (alpha * alpha)) - 1 / (1 + alpha * alpha));
}
Real ave_coef_b_phi(Real alpha) { return 4 * alpha * (1 - alpha * atan(1 / alpha)); }

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

    std::cout << "p1 = " << p1 << " pa =" << pa << " rho = " << rho << std::endl;

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
int is_ej = false;

void Mesh::InitUserMeshData(ParameterInput *pin) {
    is_ej = pin->GetOrAddReal("problem", "injection_ej", 0);
    if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
        if (static_cast<bool>(is_ej)) {
            EnrollUserBoundaryFunction(BoundaryFace::inner_x1, LoopInnerX1);
        } else {
            EnrollUserBoundaryFunction(BoundaryFace::inner_x1, Jet);
        }
    }
    return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Spherical blast wave test problem generator
//========================================================================================

Real rin = 0;

// ejetcta
Real t_ej_crit = 0.005;
Real t_ej_end = 0.015;
Real v_ej = 0.2;
Real rho_ej = 0;
Real THETA_ej = 1e-4;

// jet
Real theta_jet = 0.1;
Real t_jet_launch = 0.5;
Real t_jet_duration = 1;
Real v_jet_r = 0.8;
Real v_jet_jm = 0.4;
Real rho_jet = 0;
Real p_jet_ave = 0;

Real B_r = 0;
Real B_jm = 0;

inline void print_par(std::string name, Real value) { std::cout << name << " = " << value << std::endl; }

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    rin = pcoord->x1f(is);

    // reading hydro paramters
    Real gamma_hydro = pin->GetOrAddReal("hydro", "gamma", 1.33333333);
    Real hydro_coef = gamma_hydro / (gamma_hydro - 1);

    // reading parameters of ambient medium
    Real THETA_amb = pin->GetOrAddReal("problem", "THETA_amb", 1e-3);
    Real rho_amb = pin->GetOrAddReal("problem", "rho_amb", 1e-10);

    // reading parameters of ejecta
    THETA_ej = pin->GetOrAddReal("problem", "THETA_ej", 1e-4);
    Real M_ej = pin->GetOrAddReal("problem", "M_ej", 0.01);

    t_ej_crit = pin->GetOrAddReal("problem", "t_ej_crit", 0.005);
    t_ej_end = pin->GetOrAddReal("problem", "t_ej_duration", 0.015);
    v_ej = pin->GetOrAddReal("problem", "v_ej", 0.2);

    // reading parameters of jet
    theta_jet = pin->GetOrAddReal("problem", "theta_jet", 0.17453292519943295);
    t_jet_launch = pin->GetOrAddReal("problem", "t_jet_launch", 0.5);
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
    Real rc = 0.043333333;

    Real int_coef = 2 * PI * (0.5 + 3.0 / 8 * PI) * t_ej_crit;  // (2 * t_ej_crit - t_ej_crit * t_ej_crit / t_ej_end);
    if (static_cast<bool>(is_ej)) {
        rho_ej = M_ej / (v_ej * rin * rin * int_coef);
    } else {
        rho_ej = M_ej / (rc * rc * rc * 2 * PI * (0.5 + 3.0 / 8 * PI));
    }

    // jet calculations
    Real gamma_jet = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r - 2.0 / 3.0 * v_jet_jm * v_jet_jm);  // average gamma
    Real gamma_jet2 = gamma_jet * gamma_jet;

    Real gamma_jet_r = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r);  // gamma in radial direction
    Real h_star = Gamma_inf / gamma_jet_r;
    Real h_coef = h_star / (h_star - 1);

    Real v_dot_B = (v_jet_r * B_r + 2.0 / 3.0 * ave_coef_b_phi(Alpha) * v_jet_jm * B_jm);
    Real b2_ave = (B_r * B_r + ave_coef_b_phi(Alpha) * B_jm * B_jm) / gamma_jet / gamma_jet + v_dot_B * v_dot_B;
    Real pm_jet_ave = b2_ave / 2;

    Real e_jet = L_jet / (v_jet_r * rin * rin * 4 * PI);

    p_jet_ave = (e_jet + gamma_jet2 * v_dot_B * v_dot_B + pm_jet_ave - 2 * gamma_jet2 * h_coef * pm_jet_ave) /
                (gamma_jet2 * h_coef * hydro_coef - 1);

    Real rho_jet = (hydro_coef * p_jet_ave + 2 * pm_jet_ave) / (h_star - 1);

    Real sigma_B = pm_jet_ave / p_jet_ave;

    print_par("rho_ej", rho_ej);
    print_par("rho_jet", rho_jet);
    print_par("sigma", sigma_B);
    print_par("p_jet", p_jet_ave);
    print_par("h_star", h_star);
    print_par("gamma_r", gamma_jet_r);

    Real p_amb = rho_amb * THETA_amb;

    if (static_cast<bool>(is_ej)) {
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
    } else {
        for (int k = ks; k <= ke; k++) {
            for (int j = js; j <= je; j++) {
                for (int i = is; i <= ie; i++) {
                    Real r = pcoord->x1f(i);
                    Real sin_theta = std::sin(pcoord->x2v(j));
                    Real v = 0;
                    Real rho = 0;
                    Real p = 0;
                    if (r < rc) {
                        rho = rho_ej * (0.25 + sin_theta * sin_theta * sin_theta) * pow(r / rc, -2);
                        v = v_ej * r / rc;
                        p = rho * THETA_ej;
                    } else if (r < 4 * rc) {
                        rho = rho_ej * 0.25 * pow(r / rc, -6);
                        v = v_ej * r / rc;
                        p = rho * THETA_ej;
                    } else {
                        rho = rho_amb;
                        v = 0;
                        p = rho * THETA_amb;
                    }
                    Real g = 1.0 / sqrt(1 - v * v);
                    phydro->u(IDN, k, j, i) = g * rho;
                    phydro->u(IM1, k, j, i) = g * g * (rho + hydro_coef * p) * v;
                    phydro->u(IM2, k, j, i) = 0.0;
                    phydro->u(IM3, k, j, i) = 0.0;
                    phydro->u(IEN, k, j, i) = g * g * (rho + hydro_coef * p) - p;
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

    /* Real p_a = THETA_ej * rho_ej * (0.25 + sin(theta_jet) * sin(theta_jet) * sin(theta_jet));
     Real v1 = get_vphi(theta_jet, v_jet_jm, theta_jet);
     Real gamma_jet1 = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r - v1 * v1);
     Real B_phi1 = get_Bphi(theta_jet, B_jm, theta_jet, Alpha);
     Real v_dot_B1 = v_jet_r * B_r + v1 * B_phi1;
     Real pm_1 = ((B_r * B_r + B_phi1 * B_phi1) / gamma_jet1 / gamma_jet1 + v_dot_B1 * v_dot_B1) / 2;
     Real p1 = p_a - pm_1;
     calc_p_jet_profile(rho_jet, B_r, B_jm, v_jet_r, v_jet_jm, p_a, Alpha, theta_jet);

     bool p_positive = check_p(P);
     if (p_positive == false) {
         std::cout << "p is negative" << std::endl;
         exit(0);
     }

     Real p_ave_real = calc_p_ave(THETA, P);
     Real hh = 1 + 4 * p_ave_real / rho_jet;
     Real hh_star = hh + b2_ave / rho_jet;
     print_par("hh", hh);
     print_par("hh_star", hh_star);
     print_par("p_ave_real", p_ave_real);*/
}

size_t get_boundary_index(Coordinates *pcoord, size_t jl, size_t ju, Real theta_jet) {
    for (int j = jl; j <= ju; ++j) {
        if (pcoord->x2v(j) > theta_jet) {
            return j;
        }
    }
    return ju;
}

void Jet(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &b, Real time, Real dt, int il, int iu,
         int jl, int ju, int kl, int ku, int ngh) {
    if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") != 0) {
        std::cout << "use sphereical coordinate" << std::endl;
        exit(0);
    }

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
                            b.x2f(k, j, (il - i)) = b.x1f(k, j, (il));
                        }
                    }
                }
            }
            for (int k = kl; k <= ku + 1; ++k) {
                for (int j = jl; j <= ju; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        if (pcoord->x2v(j) < theta_jet) {
                            b.x3f(k, j, (il - i)) = get_Bphi(pcoord->x2v(j), B_jm, theta_jet, Alpha);
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
                        Real v_phi = get_vphi(pcoord->x2v(j), v_jet_jm, theta_jet);
                        Real gamma_jet = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r - v_phi * v_phi);

                        prim(IDN, k, j, il - i) = rho_jet;
                        prim(IVX, k, j, il - i) = gamma_jet * v_jet_r;
                        prim(IVY, k, j, il - i) = 0.0;
                        prim(IVZ, k, j, il - i) = gamma_jet * v_phi;
                        prim(IPR, k, j, il - i) = p_jet_ave;
                        // get_p_value(THETA, P, pcoord->x2v(j));  // p_jet_ave / (615389) * pow(10.0, p_index);
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
                    prim(IPR, k, j, il - i) = THETA_ej * prim(IDN, k, j, il - i);
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
                            b.x3f(k, j, (il - i)) = get_Bphi(pcoord->x2v(j), B_jm, theta_jet, Alpha);
                        } else {
                            b.x3f(k, j, (il - i)) = 0.0;  // b.x3f(k, j, (il));
                        }
                    }
                }
            }
        }

        for (int k = kl; k <= ku; ++k) {
            for (int j = ju; j >= jl; --j) {
                for (int i = 1; i <= ngh; ++i) {
                    if (pcoord->x2v(j) < theta_jet) {
                        Real v_phi = get_vphi(pcoord->x2v(j), v_jet_jm, theta_jet);
                        Real gamma_jet = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r - v_phi * v_phi);

                        prim(IDN, k, j, il - i) = rho_jet;
                        prim(IVX, k, j, il - i) = gamma_jet * v_jet_r;
                        prim(IVY, k, j, il - i) = 0.0;
                        prim(IVZ, k, j, il - i) = gamma_jet * v_phi;
                        prim(IPR, k, j, il - i) = p_jet_ave;
                        // get_p_value(THETA, P, pcoord->x2v(j));  // p_jet_ave / (615389) * pow(10.0, p_index);
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
