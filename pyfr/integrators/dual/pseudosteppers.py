# -*- coding: utf-8 -*-

from pyfr.integrators.dual.base import BaseDualIntegrator
import numpy as np


class BaseDualPseudoStepper(BaseDualIntegrator):
    def collect_stats(self, stats):
        super().collect_stats(stats)

        # Total number of RHS evaluations
        stats.set('solver-time-integrator', 'nfevals', self._stepper_nfevals)

        # Total number of pseudo-steps
        stats.set('solver-time-integrator', 'npseudosteps', self.npseudosteps)

    def _add_with_dts(self, *args, c):
        vals, regs = list(args[::2]), list(args[1::2])

        # Coefficients for the dual-time source term
        svals = [c*sc for sc in self._dual_time_source]

        # Normal addition
        axnpby = self._get_axnpby_kerns(len(vals))
        self._prepare_reg_banks(*regs)
        self._queue % axnpby(*vals)

        # Source addition
        axnpby = self._get_axnpby_kerns(len(svals) + 1, subdims=self._subdims)
        self._prepare_reg_banks(regs[0], self._idxcurr, *self._source_regidx)
        self._queue % axnpby(1, *svals)

    def finalise_step(self, currsoln):
        add = self._add
        pnreg = self._pseudo_stepper_nregs

        # Rotate the source registers to the right by one
        self._regidx[pnreg:] = (self._source_regidx[-1:] +
                                self._source_regidx[:-1])

        # Copy the current soln into the first source register
        add(0.0, self._regidx[pnreg], 1.0, currsoln)


class DualPseudoEulerStepper(BaseDualPseudoStepper):
    pseudo_stepper_name = 'euler'

    @property
    def _stepper_nfevals(self):
        return self.nsteps

    @property
    def _pseudo_stepper_nregs(self):
        return 2

    @property
    def _pseudo_stepper_order(self):
        return 1

    def step(self, t, dt, dtau):
        add, add_with_dts = self._add, self._add_with_dts
        rhs = self.system.rhs
        r0, r1 = self._stepper_regidx
        rat = dtau / dt

        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        rhs(t, r0, r1)
        add_with_dts(0, r1, 1, r0, dtau, r1, c=rat)

        return r1, r0


class DualPseudoTVDRK3Stepper(BaseDualPseudoStepper):
    pseudo_stepper_name = 'tvd-rk3'

    @property
    def _stepper_nfevals(self):
        return 3*self.nsteps

    @property
    def _pseudo_stepper_nregs(self):
        return 3

    @property
    def _pseudo_stepper_order(self):
        return 3

    def step(self, t, dt, dtau):
        add, add_with_dts = self._add, self._add_with_dts
        rhs = self.system.rhs
        rat = dtau / dt

        # Get the bank indices for pseudo-registers (n+1,m; n+1,m+1; rhs),
        # where m = pseudo-time and n = real-time
        r0, r1, r2 = self._stepper_regidx

        # Ensure r0 references the bank containing u(n+1,m)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # First stage;
        # r2 = -∇·f(r0); r1 = r0 + dtau*r2 - dtau*dQ/dt;
        rhs(t, r0, r2)
        add_with_dts(0, r1, 1, r0, dtau, r2, c=rat)

        # Second stage;
        # r2 = -∇·f(r1); r1 = 3/4*r0 + 1/4*r1 + 1/4*dtau*r2 - dtau/4*dQ/dt
        rhs(t, r1, r2)
        add_with_dts(1/4, r1, 3/4, r0, dtau/4, r2, c=rat/4)

        # Third stage;
        # r2 = -∇·f(r1); r1 = 1/3*r0 + 2/3*r1 + 2/3*dtau*r2 - 2/3*dtau*dQ/dt
        rhs(t, r1, r2)
        add_with_dts(2/3, r1, 1/3, r0, 2*dtau/3, r2, c=2*rat/3)

        # Return the index of the bank containing u(n+1,m+1)
        return r1, r0


class DualPseudoRK4Stepper(BaseDualPseudoStepper):
    pseudo_stepper_name = 'rk4'

    @property
    def _stepper_nfevals(self):
        return 4*self.nsteps

    @property
    def _pseudo_stepper_nregs(self):
        return 3

    @property
    def _pseudo_stepper_order(self):
        return 4

    def step(self, t, dt, dtau):
        add, add_with_dts  = self._add, self._add_with_dts
        rhs = self.system.rhs
        rat = dtau / dt

        # Get the bank indices for pseudo-registers (n+1,m; n+1,m+1; rhs),
        # where m = pseudo-time and n = real-time
        r0, r1, r2 = self._stepper_regidx

        # Ensure r0 references the bank containing u(n+1,m)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # First stage; r1 = -∇·f(r0)
        rhs(t, r0, r1)

        # Second stage; r2 = r0 + dtau/2*r1 - dtau/2*dQ/dt; r2 = -∇·f(r2)
        add_with_dts(0, r2, 1, r0, dtau/2, r1, c=rat/2)
        rhs(t, r2, r2)

        # As no subsequent stages depend on the first stage we can
        # reuse its register to start accumulating the solution with
        # r1 = r0 + dtau/6*r1 + dtau/3*r2 -(dtau/6+dtau/3)*dQ/dt
        add_with_dts(dtau/6, r1, 1, r0, dtau/3, r2, c=(1/6+1/3)*rat)

        # Third stage; here we reuse the r2 register
        # r2 = r0 + dtau/2*r2 - dtau/2*dQ/dt
        # r2 = -∇·f(r2)
        add_with_dts(dtau/2, r2, 1, r0, c=rat/2)
        rhs(t, r2, r2)

        # Accumulate; r1 = r1 + dtau/3*r2 - dtau/3*dQ/dt
        add_with_dts(1, r1, dtau/3, r2, c=rat/3)

        # Fourth stage; again we reuse r2
        # r2 = r0 + dtau*r2 - dtau*dQ/dt
        # r2 = -∇·f(r2)
        add_with_dts(dtau, r2, 1, r0, c=rat)
        rhs(t, r2, r2)

        # Final accumulation r1 = r1 + dtau/6*r2 - dtau/6*dQ/dt = u(n+1,m+1)
        add_with_dts(1, r1, dtau/6, r2, c=rat/6)

        # Return the index of the bank containing u(n+1,m+1)
        return r1, r0

class DualPseudoRK71Stepper(BaseDualPseudoStepper):
    pseudo_stepper_name = 'rk71'

    @property
    def _stepper_nfevals(self):
        return 7*self.nsteps

    @property
    def _pseudo_stepper_nregs(self):
        return 8

    @property
    def _pseudo_stepper_order(self):
        return 1

    def step(self, t, dt, dtau):
        add, add_with_dts  = self._add, self._add_with_dts
        rhs = self.system.rhs
        rat = dtau / dt

        A = np.array([[0,0,0,0,0,0,0],
            [0.2324511858361562,0,0,0,0,0,0],
            [0.2322231685703214,0.2322231685703214,0,0,0,0,0],
            [0.2322237588440728,0.231622006897831,0.2318494338899144,0,0,0,0],
            [0.2322351030333376,0.2316575500243345,0.2202846506823094,0.2208563868951234,0,0,0],
            [0.2322346239117762,0.2316802573753524,0.2207730890236098,0.2119898751137023,0.2231191886646174,0,0],
            [0.214206616344267,0.1979168835910834,0.1933813914977442,0.1536900816632531,0.0945522683179592,0.09850693266479,0]])
        B = np.array([0.2083651435574576,0.1912434615654654,0.175992229629985,0.1160474272290253,0.0721803614104373,0.0702944469173733,0.1658769296902561])
        C = np.array([0,0.2324511858361562,0.4644463371406428,0.6956951996318183,0.9050336906351049,1.119797034089058,0.9522541740790968])

        # Get the bank indices for pseudo-registers (n+1,m; n+1,m+1; rhs),
        # where m = pseudo-time and n = real-time
        r0, r1, r2, r3, r4, r5, r6, r7 = self._stepper_regidx

        # Ensure r0 references the bank containing u(n+1,m)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # First stage; r1 = -∇·f(r0)
        rhs(t, r0, r1)

        # Second stage; r2 = r0 + a10*dtau*r1 - c1*dtau*dQ/dt; r2 = -∇·f(r2)
        add_with_dts(0, r2, 1, r0, A[1,0]*dtau, r1, c=C[1]*rat)
        rhs(t, r2, r2)

        # Third stage; r3 = r0 + a20*dtau*r1 + a21*dtau*r2 - c2*dtau*dQ/dt; r3 = -∇·f(r3)
        add_with_dts(0, r3, 1, r0, A[2,0]*dtau, r1, A[2,1]*dtau, r2, c=C[2]*rat)
        rhs(t, r3, r3)

        # Fourth stage; r4 = r0 + a30*dtau*r1 + a31*dtau*r2 + a32*dtau*r3 - c3*dtau*dQ/dt; r4 = -∇·f(r4)
        add_with_dts(0, r4, 1, r0, A[3,0]*dtau, r1, A[3,1]*dtau, r2, A[3,2]*dtau, r3, c=C[3]*rat)
        rhs(t, r4, r4)

        # Fifth stage; r5 = r0 + a40*dtau*r1 + a41*dtau*r2 + a42*dtau*r3 + a43*dtau*r4 - c4*dtau*dQ/dt; r5 = -∇·f(r5)
        add_with_dts(0, r5, 1, r0, A[4,0]*dtau, r1, A[4,1]*dtau, r2, A[4,2]*dtau, r3, A[4,3]*dtau, r4, c=C[4]*rat)
        rhs(t, r5, r5)

        # Sixth stage; r6 = r0 + a50*dtau*r1 + a51*dtau*r2 + a52*dtau*r3 + a53*dtau*r4 + a54*dtau*r5 - c5*dtau*dQ/dt; r6 = -∇·f(r6)
        add_with_dts(0, r6, 1, r0, A[5,0]*dtau, r1, A[5,1]*dtau, r2, A[5,2]*dtau, r3, A[5,3]*dtau, r4, A[5,4]*dtau, r5, c=C[5]*rat)
        rhs(t, r6, r6)

        # Seventh stage; r7 = r0 + a60*dtau*r1 + a61*dtau*r2 + a62*dtau*r3 + a63*dtau*r4 + a64*dtau*r5 + a65*dtau*r6 - c6*dtau*dQ/dt; r7 = -∇·f(r7)
        add_with_dts(0, r7, 1, r0, A[6,0]*dtau, r1, A[6,1]*dtau, r2, A[6,2]*dtau, r3, A[6,3]*dtau, r4, A[6,4]*dtau, r5, A[6,5]*dtau, r6, c=C[6]*rat)
        rhs(t, r7, r7)

        # Final accumulation
        add_with_dts(B[0]*dtau, r1, 1, r0, B[1]*dtau, r2, B[2]*dtau, r3, B[3]*dtau, r4, B[4]*dtau, r5, B[5]*dtau, r6, B[6]*dtau, r7, c=rat)

        # Return the index of the bank containing u(n+1,m+1)
        return r1, r0
