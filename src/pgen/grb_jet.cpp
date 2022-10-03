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

Real rho_ej = 0;
Real rho_jet = 0;
Real rho_amb = 0;
Real gm1 = 0;

Real B_r = 0.0;
Real B_phi = 0.0;

Real rin = 1;

void Mesh::InitUserMeshData(ParameterInput *pin) {
    p_amb = pin->GetOrAddReal("problem", "p_amb", 0.0);
    rho_amb = pin->GetOrAddReal("problem", "rho_amb", 0.0);
    rin = pin->GetOrAddReal("problem", "r_in", 5e7);
    Real M_ej = pin->GetOrAddReal("problem", "M_ej", 2e31);

    v_ej = pin->GetOrAddReal("problem", "v_ej", 6e9);
    t_ej_end = pin->GetOrAddReal("problem", "t_ej_duration", 0.015);
    t_ej_crit = pin->GetOrAddReal("problem", "t_ej_crit", 0.005);
    p_ej = pin->GetOrAddReal("problem", "p_ej", 1e4);

    Real int_coef = PI * (2 * t_ej_crit - t_ej_crit * t_ej_crit / t_ej_end);  // integr
    rho_ej = M_ej / (v_ej * rin * rin * int_coef);

    theta_jet = pin->GetOrAddReal("problem", "theta_jet", 0.17453292519943295);
    t_jet_launch = pin->GetOrAddReal("problem", "t_jet_launch", 0.05);

    v_jet_r = pin->GetOrAddReal("problem", "v_jet_r", 2.4e10);
    v_jet_phi = pin->GetOrAddReal("problem", "v_jet_phi", 1.2e10);

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
    for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
            for (int i = is; i <= ie; i++) {
                Real x = pcoord->x1v(i) * std::sin(pcoord->x2v(j)) * std::cos(pcoord->x3v(k));
                Real y = pcoord->x1v(i) * std::sin(pcoord->x2v(j)) * std::sin(pcoord->x3v(k));
                Real z = pcoord->x1v(i) * std::cos(pcoord->x2v(j));
                Real rad = std::sqrt(SQR(x) + SQR(y) + SQR(z));
                // if (rad < rin * 1.01) {
                // Real sin_theta = std::sin(pcoord->x2v(j));

                // phydro->u(IDN, k, j, i) = rho_ej * (0.25 + sin_theta * sin_theta * sin_theta);
                //  std::cout << phydro->u(IDN, k, j, i) << '\n';
                // phydro->u(IM1, k, j, i) = phydro->u(IDN, k, j, i) * v_ej;
                // phydro->u(IM2, k, j, i) = 0.0;
                // phydro->u(IM3, k, j, i) = 0.0;
                // phydro->u(IEN, k, j, i) = p_ej / gm1 + 0.5 * phydro->u(IDN, k, j, i) * v_ej * v_ej;
                // } else {
                phydro->u(IDN, k, j, i) = rho_amb;
                phydro->u(IM1, k, j, i) = 0.0;
                phydro->u(IM2, k, j, i) = 0.0;
                phydro->u(IM3, k, j, i) = 0.0;
                phydro->u(IEN, k, j, i) = p_amb / gm1;
                //}
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
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    Real sin_theta = std::sin(pcoord->x2v(j));
                    prim(IDN, k, j, il - i) = rho_ej * (0.25 + sin_theta * sin_theta * sin_theta);
                    // VelProfileCyl(pco->x1v(il-i),pco->x2v(j),pco->x3v(k),v1,v2,v3);
                    prim(IVX, k, j, il - i) = v_ej;  // v1;
                    prim(IVY, k, j, il - i) = 0.0;   // v2;
                    prim(IVZ, k, j, il - i) = 0.0;   // v3;
                    prim(IPR, k, j, il - i) = p_ej;
                }
            }
        }

        if (MAGNETIC_FIELDS_ENABLED) {
            for (int k = kl; k <= ku; ++k) {
                for (int j = jl; j <= ju; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        b.x1f(k, j, (il - i)) = 0.0;  // b.x1f(k,j,il);
                    }
                }
            }
            for (int k = kl; k <= ku; ++k) {
                for (int j = jl; j <= ju + 1; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        b.x2f(k, j, (il - i)) = 0.0;  // b.x2f(k,j,il);
                    }
                }
            }
            for (int k = kl; k <= ku + 1; ++k) {
                for (int j = jl; j <= ju; ++j) {
                    for (int i = 1; i <= ngh; ++i) {
                        b.x3f(k, j, (il - i)) = 0;
                    }
                }
            }
        }
    } else if (time < t_ej_end) {
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    Real sin_theta = std::sin(pcoord->x2v(j));
                    prim(IDN, k, j, il - i) =
                        rho_ej * (0.25 + sin_theta * sin_theta * sin_theta) * t_ej_crit * t_ej_crit / time / time;

                    // VelProfileCyl(pco->x1v(il-i),pco->x2v(j),pco->x3v(k),v1,v2,v3);
                    prim(IVX, k, j, il - i) = v_ej;  // v1;
                    prim(IVY, k, j, il - i) = 0.0;
                    prim(IVZ, k, j, il - i) = 0.0;
                    prim(IPR, k, j, il - i) = p_ej;
                }
            }
        }
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
                        b.x3f(k, j, (il - i)) = 0;
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
                        b.x3f(k, j, (il - i)) = 0;
                    }
                }
            }
        }
    }

    if (time >= t_jet_launch) {
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    if (pcoord->x2v(j) < theta_jet) {
                        prim(IDN, k, j, il - i) = rho_jet;
                        prim(IVX, k, j, il - i) = v_jet_r;
                        prim(IVY, k, j, il - i) = 0.0;
                        prim(IVZ, k, j, il - i) = v_jet_phi * pcoord->x2v(j) / theta_jet;
                        prim(IPR, k, j, il - i) = p_jet;
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
                            b.x3f(k, j, (il - i)) = b.x3f(k, j, il);
                        }
                    }
                }
            }

            if (NON_BAROTROPIC_EOS) {
                for (int k = kl; k <= ku; ++k) {
                    for (int j = jl; j <= ju; ++j) {
                        for (int i = il; i <= iu; ++i) {
                            prim(IEN, k, j, i) +=
                                0.5 * (SQR(b.x3f(k, j, i)) + SQR(b.x2f(k, j, i)) + SQR(b.x1f(k, j, i)));
                        }
                    }
                }
            }
        }
    }

    // copy face-centered magnetic fields into ghost zones
}
