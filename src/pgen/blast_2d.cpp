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

Real threshold;

int RefinementCondition(MeshBlock *pmb);

void Mesh::InitUserMeshData(ParameterInput *pin) {
    if (adaptive) {
        EnrollUserRefinementCondition(RefinementCondition);
        threshold = pin->GetReal("problem", "thr");
    }

    if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(BoundaryFace::inner_x1, LoopInnerX1);
    }

    return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Spherical blast wave test problem generator
//========================================================================================

Real x0_reset, y0_reset, z0_reset;
Real ut_reset, v1_reset, uv_reset;
Real p_reset, d_reset;
Real gm1_reset;
Real rin_reset;
Real b0_ghost = 10;

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    Real rout = pin->GetReal("problem", "radius");
    Real rin = rout - pin->GetOrAddReal("problem", "ramp", 0.0);
    Real pa = pin->GetOrAddReal("problem", "pamb", 1.0);
    Real da = pin->GetOrAddReal("problem", "damb", 1.0);
    Real prat = pin->GetReal("problem", "prat");
    Real drat = pin->GetOrAddReal("problem", "drat", 1.0);
    Real v1_in = pin->GetOrAddReal("problem", "vr", 0.0);
    Real b0, angle;
    if (MAGNETIC_FIELDS_ENABLED) {
        b0 = pin->GetReal("problem", "b0");
        angle = (PI / 180.0) * pin->GetReal("problem", "angle");
    }
    Real gamma = peos->GetGamma();
    Real gm1 = gamma - 1.0;
    Real ut_in = std::sqrt(1 / (1 - v1_in * v1_in));
    Real uv_in = ut_in * v1_in;

    // get coordinates of center of blast, and convert to Cartesian if necessary
    Real x1_0 = pin->GetOrAddReal("problem", "x1_0", 0.0);
    Real x2_0 = pin->GetOrAddReal("problem", "x2_0", 0.0);
    Real x3_0 = pin->GetOrAddReal("problem", "x3_0", 0.0);
    Real x0, y0, z0;
    if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
        x0 = x1_0;
        y0 = x2_0;
        z0 = x3_0;
    } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
        x0 = x1_0 * std::cos(x2_0);
        y0 = x1_0 * std::sin(x2_0);
        z0 = x3_0;
    } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
        x0 = x1_0 * std::sin(x2_0) * std::cos(x3_0);
        y0 = x1_0 * std::sin(x2_0) * std::sin(x3_0);
        z0 = x1_0 * std::cos(x2_0);
    } else {
        // Only check legality of COORDINATE_SYSTEM once in this function
        std::stringstream msg;
        msg << "### FATAL ERROR in blast.cpp ProblemGenerator" << std::endl
            << "Unrecognized COORDINATE_SYSTEM=" << COORDINATE_SYSTEM << std::endl;
        ATHENA_ERROR(msg);
    }

    x0_reset = x0;
    y0_reset = y0;
    z0_reset = z0;
    rin_reset = rin;
    gm1_reset = gm1;
    ut_reset = ut_in;
    uv_reset = uv_in;
    v1_reset = v1_in;

    // AthenaArray<Real> bb;
    // bb.NewAthenaArray(3, ke+1, je+1, ie+1);
    // setup uniform ambient medium with spherical over-pressured region
    for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
            for (int i = is; i <= ie; i++) {
                Real rad;
                if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
                    Real x = pcoord->x1v(i);
                    Real y = pcoord->x2v(j);
                    Real z = pcoord->x3v(k);

                    rad = std::sqrt(SQR(x) + SQR(y) + SQR(z));
                } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
                    Real x = pcoord->x1v(i) * std::cos(pcoord->x2v(j));
                    Real y = pcoord->x1v(i) * std::sin(pcoord->x2v(j));
                    Real z = pcoord->x3v(k);
                    rad = std::sqrt(SQR(x) + SQR(y) + SQR(z));
                } else {  // if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0)
                    Real x = pcoord->x1v(i) * std::sin(pcoord->x2v(j)) * std::cos(pcoord->x3v(k));
                    Real y = pcoord->x1v(i) * std::sin(pcoord->x2v(j)) * std::sin(pcoord->x3v(k));
                    Real z = pcoord->x1v(i) * std::cos(pcoord->x2v(j));
                    rad = std::sqrt(SQR(x) + SQR(y) + SQR(z));
                }

                Real den = da;
                Real uv = 0.0;
                Real ut = 1.0;
                Real v1 = 0.0;

                if (fabs(rad - x0) < rout) {
                    den = drat * da / (rad / (x0 + rout)) / (rad / (x0 + rout));  // this line has been changed
                    uv = uv_in;
                    ut = ut_in;
                    v1 = v1_in;
                }

                if (rad < (x0 - rout)) {
                    den = 0.0;
                }

                phydro->u(IDN, k, j, i) = ut * den;
                if (NON_BAROTROPIC_EOS) {
                    Real pres = pa;
                    if (fabs(rad - x0) < rout) {
                        pres = prat * pa * pow(rad / (x0 + rin), -2 * (gm1 + 1));
                    }

                    if (rad < (x0 - rout)) {
                        pres = 0.0;
                    }
                    // phydro->u(IM1,k,j,i) = ut*ut*den*v1;
                    phydro->u(IM1, k, j, i) = ut * ut * (den + pres / gm1 + pres) * v1;
                    // phydro->u(IM1,k,j,i) = 0.0;
                    phydro->u(IM2, k, j, i) = 0.0;
                    phydro->u(IM3, k, j, i) = 0.0;
                    // phydro->u(IEN,k,j,i) = ut*ut*den;
                    phydro->u(IEN, k, j, i) = ut * ut * (den + pres / gm1 + pres) - pres;
                    // if (RELATIVISTIC_DYNAMICS)  // this should only ever be SR with this file
                    //       phydro->u(IEN,k,j,i) += den;
                }
            }
        }
    }
    // p_reset = pa*prat;

    // initialize interface B and total energy
    if (MAGNETIC_FIELDS_ENABLED) {
        // Initialize radial field components
        for (int k = ks; k <= ke; ++k) {
            for (int j = js; j <= je; ++j) {
                for (int i = is; i <= ie + 1; ++i) {
                    pfield->b.x1f(k, j, i) = 0;
                }
            }
        }

        // Initialize poloidal field components
        for (int k = ks; k <= ke; ++k) {
            for (int j = js; j <= je + 1; ++j) {
                for (int i = is; i <= ie; ++i) {
                    pfield->b.x2f(k, j, i) = 0;
                }
            }
        }

        // Initialize azimuthal field components
        for (int k = ks; k <= ke + 1; ++k) {
            for (int j = js; j <= je; ++j) {
                for (int i = is; i <= ie; ++i) {
                    pfield->b.x3f(k, j, i) = 0;
                }
            }
        }
    }
}

void LoopInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b, Real time, Real dt, int il,
                 int iu, int jl, int ju, int kl, int ku, int ngh) {
    // Real rad, phi, z;
    Real v1, v2, v3;
    for (int k = kl; k <= ku; ++k) {
        for (int j = jl; j <= ju; ++j) {
            for (int i = 1; i <= ngh; ++i) {
                if (i == 1) {
                    prim(IDN, k, j, il - i) = 1.0;  // d_reset;//rho0;
                } else {
                    prim(IDN, k, j, il - i) = 0.0;
                }
                // VelProfileCyl(pco->x1v(il-i),pco->x2v(j),pco->x3v(k),v1,v2,v3);
                if (i == 1) {
                    prim(IM1, k, j, il - i) = 0.1;  // v1;
                } else {
                    prim(IM1, k, j, il - i) = 0.0;  // v1;
                }
                prim(IM2, k, j, il - i) = 0.0;  // v2;
                prim(IM3, k, j, il - i) = 0.0;  // v3;
                // if (NON_BAROTROPIC_EOS)
                if (i == 1) {
                    prim(IEN, k, j, il - i) = prim(IDN, k, j, il - i);  // isocs2*prim(IDN,k,j,il-i);
                } else {
                    prim(IEN, k, j, il - i) = prim(IDN, k, j, il - i);  // isocs2*prim(IDN,k,j,il-i);
                }
            }
        }
    }
    // copy face-centered magnetic fields into ghost zones
    if (MAGNETIC_FIELDS_ENABLED) {
        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    b.x1f(k, j, (il - i)) = 0;
                    // b.x1f(k, j, il);
                }
            }
        }

        for (int k = kl; k <= ku; ++k) {
            for (int j = jl; j <= ju + 1; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    b.x2f(k, j, (il - i)) = 0;
                    // b.x2f(k, j, il);
                }
            }
        }

        for (int k = kl; k <= ku + 1; ++k) {
            for (int j = jl; j <= ju; ++j) {
                for (int i = 1; i <= ngh; ++i) {
                    if (i == 1) {
                        b.x3f(k, j, (il - i)) = b0_ghost;  // b.x3f(k,j,il);
                    } else {
                        b.x3f(k, j, (il - i)) = 0;
                    }
                }
            }
        }
    }
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//! \brief Check radius of sphere to make sure it is round
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
    if (!pin->GetOrAddBoolean("problem", "compute_error", false)) return;
    MeshBlock *pmb = my_blocks(0);

    // analysis - check shape of the spherical blast wave
    int is = pmb->is, ie = pmb->ie;
    int js = pmb->js, je = pmb->je;
    int ks = pmb->ks, ke = pmb->ke;
    AthenaArray<Real> pr;
    pr.InitWithShallowSlice(pmb->phydro->w, 4, IPR, 1);

    // get coordinate location of the center, convert to Cartesian
    Real x1_0 = pin->GetOrAddReal("problem", "x1_0", 0.0);
    Real x2_0 = pin->GetOrAddReal("problem", "x2_0", 0.0);
    Real x3_0 = pin->GetOrAddReal("problem", "x3_0", 0.0);
    Real x0, y0, z0;
    if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
        x0 = x1_0;
        y0 = x2_0;
        z0 = x3_0;
    } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
        x0 = x1_0 * std::cos(x2_0);
        y0 = x1_0 * std::sin(x2_0);
        z0 = x3_0;
    } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
        x0 = x1_0 * std::sin(x2_0) * std::cos(x3_0);
        y0 = x1_0 * std::sin(x2_0) * std::sin(x3_0);
        z0 = x1_0 * std::cos(x2_0);
    } else {
        // Only check legality of COORDINATE_SYSTEM once in this function
        std::stringstream msg;
        msg << "### FATAL ERROR in blast.cpp ParameterInput" << std::endl
            << "Unrecognized COORDINATE_SYSTEM= " << COORDINATE_SYSTEM << std::endl;
        ATHENA_ERROR(msg);
    }

    // find indices of the center
    int ic, jc, kc;
    for (ic = is; ic <= ie; ic++)
        if (pmb->pcoord->x1f(ic) > x1_0) break;
    ic--;
    for (jc = pmb->js; jc <= pmb->je; jc++)
        if (pmb->pcoord->x2f(jc) > x2_0) break;
    jc--;
    for (kc = pmb->ks; kc <= pmb->ke; kc++)
        if (pmb->pcoord->x3f(kc) > x3_0) break;
    kc--;

    // search pressure maximum in each direction
    Real rmax = 0.0, rmin = 100.0, rave = 0.0;
    int nr = 0;
    for (int o = 0; o <= 6; o++) {
        int ios = 0, jos = 0, kos = 0;
        if (o == 1)
            ios = -10;
        else if (o == 2)
            ios = 10;
        else if (o == 3)
            jos = -10;
        else if (o == 4)
            jos = 10;
        else if (o == 5)
            kos = -10;
        else if (o == 6)
            kos = 10;
        for (int d = 0; d < 6; d++) {
            Real pmax = 0.0;
            int imax(0), jmax(0), kmax(0);
            if (d == 0) {
                if (ios != 0) continue;
                jmax = jc + jos, kmax = kc + kos;
                for (int i = ic; i >= is; i--) {
                    if (pr(kmax, jmax, i) > pmax) {
                        pmax = pr(kmax, jmax, i);
                        imax = i;
                    }
                }
            } else if (d == 1) {
                if (ios != 0) continue;
                jmax = jc + jos, kmax = kc + kos;
                for (int i = ic; i <= ie; i++) {
                    if (pr(kmax, jmax, i) > pmax) {
                        pmax = pr(kmax, jmax, i);
                        imax = i;
                    }
                }
            } else if (d == 2) {
                if (jos != 0) continue;
                imax = ic + ios, kmax = kc + kos;
                for (int j = jc; j >= js; j--) {
                    if (pr(kmax, j, imax) > pmax) {
                        pmax = pr(kmax, j, imax);
                        jmax = j;
                    }
                }
            } else if (d == 3) {
                if (jos != 0) continue;
                imax = ic + ios, kmax = kc + kos;
                for (int j = jc; j <= je; j++) {
                    if (pr(kmax, j, imax) > pmax) {
                        pmax = pr(kmax, j, imax);
                        jmax = j;
                    }
                }
            } else if (d == 4) {
                if (kos != 0) continue;
                imax = ic + ios, jmax = jc + jos;
                for (int k = kc; k >= ks; k--) {
                    if (pr(k, jmax, imax) > pmax) {
                        pmax = pr(k, jmax, imax);
                        kmax = k;
                    }
                }
            } else {  // if (d == 5) {
                if (kos != 0) continue;
                imax = ic + ios, jmax = jc + jos;
                for (int k = kc; k <= ke; k++) {
                    if (pr(k, jmax, imax) > pmax) {
                        pmax = pr(k, jmax, imax);
                        kmax = k;
                    }
                }
            }

            Real xm, ym, zm;
            Real x1m = pmb->pcoord->x1v(imax);
            Real x2m = pmb->pcoord->x2v(jmax);
            Real x3m = pmb->pcoord->x3v(kmax);
            if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
                xm = x1m;
                ym = x2m;
                zm = x3m;
            } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
                xm = x1m * std::cos(x2m);
                ym = x1m * std::sin(x2m);
                zm = x3m;
            } else {  // if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
                xm = x1m * std::sin(x2m) * std::cos(x3m);
                ym = x1m * std::sin(x2m) * std::sin(x3m);
                zm = x1m * std::cos(x2m);
            }
            Real rad = std::sqrt(SQR(xm - x0) + SQR(ym - y0) + SQR(zm - z0));
            if (rad > rmax) rmax = rad;
            if (rad < rmin) rmin = rad;
            rave += rad;
            nr++;
        }
    }
    rave /= static_cast<Real>(nr);

    // use physical grid spacing at center of blast
    Real dr_max;
    Real x1c = pmb->pcoord->x1v(ic);
    Real dx1c = pmb->pcoord->dx1f(ic);
    Real x2c = pmb->pcoord->x2v(jc);
    Real dx2c = pmb->pcoord->dx2f(jc);
    Real dx3c = pmb->pcoord->dx3f(kc);
    if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
        dr_max = std::max(std::max(dx1c, dx2c), dx3c);
    } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
        dr_max = std::max(std::max(dx1c, x1c * dx2c), dx3c);
    } else {  // if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
        dr_max = std::max(std::max(dx1c, x1c * dx2c), x1c * std::sin(x2c) * dx3c);
    }
    Real deform = (rmax - rmin) / dr_max;

    // only the root process outputs the data
    if (Globals::my_rank == 0) {
        std::string fname;
        fname.assign("blastwave-shape.dat");
        std::stringstream msg;
        FILE *pfile;

        // The file exists -- reopen the file in append mode
        if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
            if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
                msg << "### FATAL ERROR in function [Mesh::UserWorkAfterLoop]" << std::endl
                    << "Blast shape output file could not be opened" << std::endl;
                ATHENA_ERROR(msg);
            }

            // The file does not exist -- open the file in write mode and add headers
        } else {
            if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
                msg << "### FATAL ERROR in function [Mesh::UserWorkAfterLoop]" << std::endl
                    << "Blast shape output file could not be opened" << std::endl;
                ATHENA_ERROR(msg);
            }
        }
        std::fprintf(pfile, "# Offset blast wave test in %s coordinates:\n", COORDINATE_SYSTEM);
        std::fprintf(pfile, "# Rmax       Rmin       Rave        Deformation\n");
        std::fprintf(pfile, "%e  %e  %e  %e \n", rmax, rmin, rave, deform);
        std::fclose(pfile);
    }
    return;
}

// refinement condition: check the maximum pressure gradient
int RefinementCondition(MeshBlock *pmb) {
    AthenaArray<Real> &w = pmb->phydro->w;
    Real maxeps = 0.0;
    if (pmb->pmy_mesh->f3) {
        for (int k = pmb->ks - 1; k <= pmb->ke + 1; k++) {
            for (int j = pmb->js - 1; j <= pmb->je + 1; j++) {
                for (int i = pmb->is - 1; i <= pmb->ie + 1; i++) {
                    Real eps = std::sqrt(SQR(0.5 * (w(IPR, k, j, i + 1) - w(IPR, k, j, i - 1))) +
                                         SQR(0.5 * (w(IPR, k, j + 1, i) - w(IPR, k, j - 1, i))) +
                                         SQR(0.5 * (w(IPR, k + 1, j, i) - w(IPR, k - 1, j, i)))) /
                               w(IPR, k, j, i);
                    maxeps = std::max(maxeps, eps);
                }
            }
        }
    } else if (pmb->pmy_mesh->f2) {
        int k = pmb->ks;
        for (int j = pmb->js - 1; j <= pmb->je + 1; j++) {
            for (int i = pmb->is - 1; i <= pmb->ie + 1; i++) {
                Real eps = std::sqrt(SQR(0.5 * (w(IPR, k, j, i + 1) - w(IPR, k, j, i - 1))) +
                                     SQR(0.5 * (w(IPR, k, j + 1, i) - w(IPR, k, j - 1, i)))) /
                           w(IPR, k, j, i);
                maxeps = std::max(maxeps, eps);
            }
        }
    } else {
        return 0;
    }

    if (maxeps > threshold) return 1;
    if (maxeps < 0.25 * threshold) return -1;
    return 0;
}
