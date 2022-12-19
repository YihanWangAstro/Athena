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

namespace jet1 {
    const Real const_jet_alpha = 0.5;
    size_t data_size = 10000;
    std::vector<Real> THETA;
    std::vector<Real> P;

    inline Real Bphi(Real theta, Real Bjm, Real theta_j) {
        const Real theta_m = const_jet_alpha * theta_j;
        if (theta < theta_m) {
            return Bjm * theta / theta_m;
        } else if (theta <= theta_j) {
            return Bjm * theta_m / theta;
        } else {
            return 0;
        }
    }

    inline Real dBdtheta(Real theta, Real Bjm, Real theta_j) {
        const Real theta_m = const_jet_alpha * theta_j;
        if (theta < theta_m) {
            return Bjm / theta_m;
        } else if (theta <= theta_j) {
            return -Bjm * theta_m / theta / theta;
        } else {
            return 0;
        }
    }

    inline Real vphi(Real theta, Real vjm, Real theta_j) {
        const Real theta_m = const_jet_alpha * theta_j;
        if (theta < theta_m) {
            return vjm * theta / theta_m;
        } else if (theta <= theta_j) {
            return vjm * theta_m / theta;
        } else {
            return 0;
        }
    }

    inline Real dvdtheta(Real theta, Real vjm, Real theta_j) {
        const Real theta_m = const_jet_alpha * theta_j;
        if (theta < theta_m) {
            return vjm / theta_m;
        } else if (theta <= theta_j) {
            return -vjm * theta_m / theta / theta;
        } else {
            return 0;
        }
    }

    Real dpdtheta(Real theta, Real rho, Real p_now, Real Bjm, Real vr, Real vjm, Real theta_j) {
        Real Bphi_ = Bphi(theta, Bjm, theta_j);
        Real vphi_ = vphi(theta, vjm, theta_j);
        Real dBdtheta_ = dBdtheta(theta, Bjm, theta_j);
        Real dvdtheta_ = dvdtheta(theta, vjm, theta_j);

        Real g = 1.0 / std::sqrt(1 - vr * vr - vphi_ * vphi_);
        Real g2 = g * g;
        Real b2 = Bphi_ * Bphi_ / g / g + vphi_ * Bphi_ * vphi_ * Bphi_;

        Real rhow = rho + 4 * p_now + b2;
        // std::cout << theta << ' ' << Bphi_ << ' ' << vphi_ << ' ' << rho << ' ' << p_now << ' ' << b2 << ' ' << rhow
        //          << ' ' << dBdtheta_ << ' ' << dvdtheta_ << '\n';
        return (rhow * g2 * vphi_ * vphi_ - Bphi_ * Bphi_ / g2) / theta - Bphi_ * dBdtheta_ / g2 -
               (vphi_ * Bphi_) * (Bphi_ * dvdtheta_ + vphi_ * dBdtheta_);
    }

    void calc_p_jet_profile(Real rho, Real Bjm, Real vr, Real vjm, Real pa, Real theta_j) {
        Real Bphi1 = Bphi(theta_j, Bjm, theta_j);
        Real vphi1 = vphi(theta_j, vjm, theta_j);
        Real gg = 1 / sqrt(1 - vphi1 * vphi1 - vr * vr);
        Real p1 = pa;
        //-(Bphi1 * Bphi1 / gg / gg + vphi1 * Bphi1 * vphi1 * Bphi1) / 2;
        // std::cout << Bphi1 << ' ' << gg << ' ' << vphi1 << '\n';

        THETA.resize(data_size + 1);
        P.resize(data_size + 1);
        size_t len = THETA.size() - 1;

        // std::cout << "p1 = " << p1 << " pa =" << pa << " rho = " << rho << std::endl;

        for (size_t i = 0; i <= len; i++) {
            THETA[i] = theta_j * static_cast<Real>(i) / static_cast<Real>(len);
            P[i] = p1;
        }
        Real dtheta = THETA[1] - THETA[0];
        for (int i = data_size - 1; i >= 0; i--) {
            P[i] = P[i + 1] - dpdtheta(THETA[i + 1], rho, P[i + 1], Bjm, vr, vjm, theta_j) * dtheta;
            // if (P[i] < 0) P[i] = 0;
        }
    }
}  // namespace jet1

namespace jet2 {
    const Real const_jet_alpha = 0.4;
    size_t data_size = 10000;
    std::vector<Real> THETA;
    std::vector<Real> P;

    inline Real Bphi(Real theta, Real Bjm, Real theta_j) {
        Real theta_m = theta_j * const_jet_alpha;
        Real ratio = theta / theta_m;
        return 2 * Bjm * ratio / (1 + ratio * ratio);
    }

    inline Real dBdtheta(Real theta, Real Bjm, Real theta_j) {
        Real theta_m = theta_j * const_jet_alpha;
        Real ratio = theta / theta_m;
        return 2 * Bjm * (1 - ratio * ratio) / (1 + ratio * ratio) / (1 + ratio * ratio) / theta_m;
    }

    inline Real vphi(Real theta, Real vjm, Real theta_j) {
        Real ratio = theta / theta_j;
        return vjm * ratio;
    }

    inline Real dvdtheta(Real theta, Real vjm, Real theta_j) { return vjm / theta_j; }

    Real dpdtheta(Real theta, Real rho, Real p_now, Real Bjm, Real vr, Real vjm, Real theta_j) {
        Real Bphi_ = Bphi(theta, Bjm, theta_j);
        Real vphi_ = vphi(theta, vjm, theta_j);
        Real dBdtheta_ = dBdtheta(theta, Bjm, theta_j);
        Real dvdtheta_ = dvdtheta(theta, vjm, theta_j);

        Real g = 1.0 / std::sqrt(1 - vr * vr - vphi_ * vphi_);
        Real g2 = g * g;
        Real b2 = Bphi_ * Bphi_ / g / g + vphi_ * Bphi_ * vphi_ * Bphi_;

        Real rhow = rho + 4 * p_now + b2;
        return (rhow * g2 * vphi_ * vphi_ - Bphi_ * Bphi_ / g2) / theta - Bphi_ * dBdtheta_ / g2 -
               (vphi_ * Bphi_) * (Bphi_ * dvdtheta_ + vphi_ * dBdtheta_);
    }

    void calc_p_jet_profile(Real rho, Real Bjm, Real vr, Real vjm, Real pa, Real theta_j) {
        Real Bphi1 = Bphi(theta_j, Bjm, theta_j);
        Real vphi1 = vphi(theta_j, vjm, theta_j);
        Real gg = 1 / sqrt(1 - vphi1 * vphi1 - vr * vr);
        Real p1 = pa;
        //    -(Bphi1 * Bphi1 / gg / gg + vphi1 * Bphi1 * vphi1 * Bphi1) / 2;

        THETA.resize(data_size + 1);
        P.resize(data_size + 1);
        size_t len = THETA.size() - 1;

        // std::cout << "p1 = " << p1 << " pa =" << pa << " rho = " << rho << std::endl;

        for (size_t i = 0; i <= len; i++) {
            THETA[i] = theta_j * static_cast<Real>(i) / static_cast<Real>(len);
            P[i] = p1;
        }
        Real dtheta = THETA[1] - THETA[0];
        for (int i = data_size - 1; i >= 0; i--) {
            P[i] = P[i + 1] - dpdtheta(THETA[i + 1], rho, P[i + 1], Bjm, vr, vjm, theta_j) * dtheta;
            // if (P[i] < 0) P[i] = 0;
        }
    }
}  // namespace jet2

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

void print_profile(std::vector<Real> &theta, std::vector<Real> &p) {
    for (size_t i = 0; i < p.size(); i += 100) {
        std::cout << "theta = " << theta[i] << ", p = " << p[i] << std::endl;
    }
}

bool check_p(std::vector<Real> &p) {
    for (auto pi : p) {
        if (pi < 0) {
            return false;
        }
    }
    return true;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Spherical blast wave test problem generator
//========================================================================================

Real hydro_coef = 4;
Real p_amb = 1e-16;
Real rho_amb = 1.35e-3;
Real gamma_hydro = 1.33333333333333333;

// ejetcta
Real v_ej = 0.2;
Real rho_ej = 0;
Real rho_tail = 0;
Real k_ej = 1e-6;
Real r_c = 0.04333333;
Real tail_n = 8;

// jet
Real theta_jet = 0.1;
Real t_jet_duration = 1;
Real v_jet_r = 0.8;
Real v_jet_jm = 0.4;
Real rho_jet = 0;
Real p_jet = 0;

Real B_jm = 0;

Real t_wind_launch = 0;
Real B_wind = 0;
Real rho_wind = 0;
Real v_wind = 0.1;
Real p_wind = 0;

int jet_model = 0;
inline void print_par(std::string name, Real value) { std::cout << name << " = " << value << std::endl; }

void Mesh::InitUserMeshData(ParameterInput *pin) {
    // reading model parameters
    jet_model = pin->GetOrAddReal("problem", "jet_model", 0);

    // reading coordinates
    Real rin = pin->GetOrAddReal("problem", "r_in", 1.6666666666e-3);
    // reading hydro paramters
    gamma_hydro = pin->GetOrAddReal("hydro", "gamma", 1.33333333);
    hydro_coef = gamma_hydro / (gamma_hydro - 1);

    // reading parameters of ambient medium
    p_amb = pin->GetOrAddReal("problem", "p_amb", 1e-16);
    rho_amb = pin->GetOrAddReal("problem", "rho_amb", 1.35e-6);

    // reading parameters of ejecta
    k_ej = pin->GetOrAddReal("problem", "k_ej", 1e-6);
    Real M_ej = pin->GetOrAddReal("problem", "M_ej", 0.01);
    v_ej = pin->GetOrAddReal("problem", "v_ej", 0.2);
    r_c = pin->GetOrAddReal("problem", "r_c", 0.043333);

    // reading parameters of jet
    theta_jet = pin->GetOrAddReal("problem", "theta_jet", 0.17453292519943295);
    t_jet_duration = pin->GetOrAddReal("problem", "t_jet_duration", 1);
    v_jet_r = pin->GetOrAddReal("problem", "v_jet_r", 0.8);
    v_jet_jm = pin->GetOrAddReal("problem", "v_jet_jm", 0.4);
    Real Gamma_inf = pin->GetOrAddReal("problem", "Gamma_inf", 300);
    Real L_jet = pin->GetOrAddReal("problem", "L_jet", 0.00278);

    B_jm = pin->GetOrAddReal("problem", "B_jm", 6.777);
    // Real sigma_B = pin->GetOrAddReal("problem", "sigma_B", 1);

    // reading parameters of wind
    t_wind_launch = pin->GetOrAddReal("problem", "t_wind_launch", 0.5);
    B_wind = pin->GetOrAddReal("problem", "B_mag", 0.1);

    /// initializing variables
    Real gamma_wind = 1 / sqrt(1 - v_wind * v_wind);
    Real b2_wind = B_wind * B_wind / gamma_wind / gamma_wind;
    Real sigma_wind = 100;

    rho_wind = b2_wind / sigma_wind;

    Real L_wind = gamma_wind * gamma_wind * (rho_wind + b2_wind) * 4 * PI * rin * rin * v_wind;

    p_wind = rho_wind * 1e-6;
    // ejecta calculations

    rho_ej = M_ej / (r_c * r_c * r_c * 2 * PI * (0.5 + 3.0 / 8 * PI));
    rho_tail = 0.01 * M_ej / (4 * PI * r_c * r_c * r_c * (pow(4, 3 - tail_n) - 1) / (3 - tail_n));

    // jet calculations

    Real gamma_jet_r = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r);  // gamma in radial direction

    Real w = Gamma_inf / gamma_jet_r;

    Real e_jet = L_jet / (v_jet_r * rin * rin * 4 * PI);

    rho_jet = e_jet / gamma_jet_r / gamma_jet_r / w;

    Real b2 = 0.716 * (B_jm * B_jm) / gamma_jet_r / gamma_jet_r + 0.454 * (v_jet_jm * B_jm * v_jet_jm * B_jm);

    Real eta = w - b2 / rho_jet;

    Real sigma = w / eta - 1;

    p_jet = rho_jet * (eta - 1) * (gamma_hydro - 1) / gamma_hydro;

    print_par("rho_ej", rho_ej);
    print_par("rho_jet", rho_jet);
    print_par("rho_tail", rho_tail);
    print_par("sigma", sigma);
    print_par("p_jet", p_jet);
    print_par("w", w);
    print_par("eta", eta);
    print_par("p_amb", p_amb);
    print_par("p_ej_crit", k_ej * pow(rho_ej, gamma_hydro));
    print_par("eta_ej_crit", gamma_hydro / (gamma_hydro - 1) * k_ej * pow(rho_ej, gamma_hydro) / rho_ej + 1);
    print_par("p_ej_inj", k_ej * pow(rho_ej * r_c * r_c / rin / rin, gamma_hydro));
    print_par("rho_wind", rho_wind);
    print_par("p_wind", p_wind);
    print_par("L_wind", L_wind * 1.8e54 * 4 * PI);

    if (jet_model == 2) {
        Real p_a = p_jet;
        for (int i = 0; i < 100; i++) {
            jet2::calc_p_jet_profile(rho_jet, B_jm, v_jet_r, v_jet_jm, p_a, theta_jet);
            Real p_ave_real = calc_p_ave(jet2::THETA, jet2::P);
            Real rtol = fabs(p_ave_real - p_jet) / p_jet;

            if (rtol < 0.01) {
                break;
            }
            if (p_ave_real < p_jet) {
                p_a *= (1 + rtol * 0.5);
            } else {
                p_a *= (1 - rtol * 0.5);
            }
        }
        print_profile(jet2::THETA, jet2::P);
        Real p_ave_real = calc_p_ave(jet2::THETA, jet2::P);
        Real hh = 1 + gamma_hydro / (gamma_hydro - 1) * p_ave_real / rho_jet;
        Real hh_star = hh + b2 / rho_jet;
        print_par("eta", hh);
        print_par("w", hh_star);
        print_par("p_jet<from profile>", p_ave_real);
    } else if (jet_model == 1) {
        Real p_a = p_jet;
        for (int i = 0; i < 100; i++) {
            jet1::calc_p_jet_profile(rho_jet, B_jm, v_jet_r, v_jet_jm, p_a, theta_jet);
            Real p_ave_real = calc_p_ave(jet1::THETA, jet1::P);
            Real rtol = fabs(p_ave_real - p_jet) / p_jet;

            if (rtol < 0.01) {
                break;
            }
            if (p_ave_real < p_jet) {
                p_a *= (1 + rtol * 0.5);
            } else {
                p_a *= (1 - rtol * 0.5);
            }
        }
        print_profile(jet1::THETA, jet1::P);
        Real p_ave_real = calc_p_ave(jet1::THETA, jet1::P);
        Real hh = 1 + gamma_hydro / (gamma_hydro - 1) * p_ave_real / rho_jet;
        Real hh_star = hh + b2 / rho_jet;
        print_par("eta", hh);
        print_par("w", hh_star);
        print_par("p_jet<from profile>", p_ave_real);
    }

    if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(BoundaryFace::inner_x1, LoopInnerX1);
    }
    return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
            for (int i = is; i <= ie; i++) {
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

    if (time < t_jet_duration) {
        // std::cout << "jet launch t= " << time << "\n";
        if (MAGNETIC_FIELDS_ENABLED) {
            for (int k = kl; k <= ku; ++k) {
                for (int j = jl; j <= ju; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        if (pcoord->x2v(j) < theta_jet) {
                            b.x1f(k, j, (il - i)) = 0.0;
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
                            if (jet_model == 0) {
                                b.x3f(k, j, (il - i)) = B_jm;
                            } else if (jet_model == 1) {
                                b.x3f(k, j, (il - i)) = jet1::Bphi(pcoord->x2v(j), B_jm, theta_jet);
                            } else if (jet_model == 2) {
                                b.x3f(k, j, (il - i)) = jet2::Bphi(pcoord->x2v(j), B_jm, theta_jet);
                            } else {
                                b.x3f(k, j, (il - i)) = 0.0;
                            }
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
                        if (jet_model == 0) {
                            Real v_phi = pcoord->x2v(j) / theta_jet * v_jet_jm;
                            Real gamma_jet = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r - v_phi * v_phi);
                            prim(IDN, k, j, il - i) = rho_jet;
                            prim(IVX, k, j, il - i) = gamma_jet * v_jet_r;
                            prim(IVY, k, j, il - i) = 0.0;
                            prim(IVZ, k, j, il - i) = gamma_jet * v_phi;
                            prim(IPR, k, j, il - i) = p_jet;
                        } else if (jet_model == 1) {
                            Real v_phi = jet1::vphi(pcoord->x2v(j), v_jet_jm, theta_jet);
                            Real gamma_jet = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r - v_phi * v_phi);

                            prim(IDN, k, j, il - i) = rho_jet;
                            prim(IVX, k, j, il - i) = gamma_jet * v_jet_r;
                            prim(IVY, k, j, il - i) = 0.0;
                            prim(IVZ, k, j, il - i) = gamma_jet * v_phi;
                            prim(IPR, k, j, il - i) = get_p_value(jet1::THETA, jet1::P, pcoord->x2v(j));
                        } else if (jet_model == 2) {
                            Real v_phi = jet2::vphi(pcoord->x2v(j), v_jet_jm, theta_jet);
                            Real gamma_jet = 1.0 / sqrt(1.0 - v_jet_r * v_jet_r - v_phi * v_phi);

                            prim(IDN, k, j, il - i) = rho_jet;
                            prim(IVX, k, j, il - i) = gamma_jet * v_jet_r;
                            prim(IVY, k, j, il - i) = 0.0;
                            prim(IVZ, k, j, il - i) = gamma_jet * v_phi;
                            prim(IPR, k, j, il - i) = get_p_value(jet2::THETA, jet2::P, pcoord->x2v(j));
                        } else {
                            prim(IDN, k, j, il - i) = prim(IDN, k, j, il);
                            prim(IVX, k, j, il - i) = prim(IVX, k, j, il);
                            prim(IVY, k, j, il - i) = prim(IVY, k, j, il);
                            prim(IVZ, k, j, il - i) = prim(IVZ, k, j, il);
                            prim(IPR, k, j, il - i) = prim(IPR, k, j, il);
                        }
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
        // std::cout << "jet end t= " << time << "\n";
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
                        b.x3f(k, j, (il - i)) = B_wind;
                    }
                }
            }
        }
    }

    // copy face-centered magnetic fields into ghost zones
}
