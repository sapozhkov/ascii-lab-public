"""Inertial departure after the loaded bumblebee has cleared the petals."""
from math import atan2, cos, exp, sin

import numpy as np

from foraging import Forager, angle_error, bend_position, limit
from lotus_weather import sample_patch


class BankedForager(Forager):
    def departure_solver(self, garden, start, velocity, thrust, yaw_rate):
        dt = 1 / self.rate
        position = self.body[start].copy()
        yaw, pitch, grip, tongue = self.attitude[start]
        foothold = self.uv[start].copy()
        roll = roll_rate = 0.
        self.roll = np.zeros(len(self.body))
        # The open corridor leads towards the far left bank. Back away until
        # the head lines up with the corridor, then spend more thrust on speed.
        outbound = np.array([-0.97, 0.25])
        outbound /= np.linalg.norm(outbound)
        heading = atan2(outbound[1], outbound[0])
        state = 'orientation'
        for i in range(start + 1, len(self.body)):
            t = i * dt
            air = garden.airflow(t, position)
            forward = np.array([cos(yaw), sin(yaw)])
            side = np.array([-forward[1], forward[0]])
            alignment = np.clip((1. + forward @ outbound) / 2., 0., 1.)
            speed = 158. + 165. * alignment**2
            desired = np.r_[speed * outbound, 75.] + .25 * air

            # Carry momentum through the turn. The limited lateral component
            # of the wing force bends the velocity vector over a finite radius.
            drag = .47 * (velocity - air)
            drive = (desired - velocity) * 2.8 + .47 * (desired - air)
            drive[:2] = limit(drive[:2], 286.)
            drive[2] = np.clip(drive[2], -145., 208.)
            thrust += (drive - thrust) * (1. - exp(-dt / .23))
            acceleration = thrust - drag
            velocity += dt * acceleration
            position += dt * velocity

            # Differential wing force supplies bounded yaw torque. Rotational
            # drag arrests the heavy body as it settles on the exit heading.
            error = angle_error(heading, yaw)
            torque = np.clip(5.0 * error - 1.9 * yaw_rate, -3.6, 3.6)
            yaw_rate += (torque - 1.65 * yaw_rate) * dt
            yaw += yaw_rate * dt
            commanded_roll = np.clip(atan2(thrust[:2] @ side, 550.), -.43, .43)
            roll_rate += (21. * (commanded_roll-roll) - 7.8 * roll_rate) * dt
            roll += dt * roll_rate
            trim = np.clip(-.0013 * (forward @ acceleration[:2]) + .0011 * velocity[2], -.31, .30)
            pitch += (trim - pitch) * (1. - exp(-dt / .21))
            grip *= exp(-dt / .062)
            tongue *= exp(-dt / .14)

            if state == 'orientation' and alignment > .80:
                state = 'depart'
                self.events['depart'] = t
            foot = sample_patch(garden.tissue(10,t)[0], *foothold)
            self.body[i] = position
            self.reference[i] = bend_position(garden, foot, t, 1)
            self.uv[i] = foothold
            self.attitude[i] = yaw, pitch, grip, tongue
            self.roll[i] = roll
            self.states.append(state)

    def pose(self, garden, t):
        pose = super().pose(garden, t)
        i = round((t % self.end) * self.rate)
        return pose._replace(bank=self.roll[i])
