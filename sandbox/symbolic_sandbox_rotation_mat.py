#!/usr/bin/env python3

import sympy as sp

y, p, r = sp.symbols('y p r') # yaw, pitch, roll
Rz = sp.Matrix([[sp.cos(y), -sp.sin(y), 0],
                [sp.sin(y),  sp.cos(y), 0],
                [        0,          0, 1]])
Ry = sp.Matrix([[ sp.cos(p), 0, sp.sin(p)],
                [         0, 1,         0],
                [-sp.sin(p), 0, sp.cos(p)]])
Rx = sp.Matrix([[1,         0,          0],
                [0, sp.cos(r), -sp.sin(r)],
                [0, sp.sin(r),  sp.cos(r)]])
R = Ry*Rz*Rx
sp.pprint(R)

atan = sp.atan2(-R[2,0],R[0,0])
sp.pprint(sp.simplify(atan))
