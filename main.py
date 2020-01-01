import plyfile
import torch

R = torch.ones(3, 1) * 0.00000000001 # in axis/angle form
R.requires_grad_()

def vector_size(v):
    return (v * v).sum(axis=0).sqrt()

def rotate_vertex(v, R):
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    theta = vector_size(R)
    k = R / theta
    a = v * theta.cos()
    b = torch.cross(k, v) * theta.sin()
    c = k * (k * v).sum() * (1 - theta.cos())
    return a + b + c

def rotate_face(f, R):
    return torch.stack([
        rotate_vertex(f[0], R),
        rotate_vertex(f[1], R),
        rotate_vertex(f[2], R)
    ])

def face_area(f):
    # a sanity check to make sure face rotation works properly!
    # via Heron's formula
    a = vector_size(f[0] - f[1])
    b = vector_size(f[1] - f[2])
    c = vector_size(f[2] - f[0])
    s = (a + b + c) / 2

    return (s * (s - a) * (s - b) * (s - c)).sqrt()





plydata = plyfile.PlyData.read('bunny/reconstruction/bun_zipper_res4.ply')

vertices = []
for vertex in plydata['vertex'].data:
    vertices.append(torch.tensor([[vertex['x']], [vertex['y']], [vertex['z']]]))

faces = []
for (face,) in plydata['face'].data:
    a = vertices[face[0]]
    b = vertices[face[1]]
    c = vertices[face[2]]
    faces.append(torch.stack([a, b, c]))

C_TRAV = 1/80   # from PBRT kdtreeaccel.cpp:435
C_ISECT = 1 # from PBRT kdtreeaccel.cpp:435

minf = float('-inf')
maxf = float('+inf')








def bounding_box_area(o, c):
    # just for fun, 2(ab + bc + cd) = (a + b + c)^2 - (a^2 + b^2 + c^2)
    extent = c - o
    return extent.sum() * extent.sum() - (extent * extent).sum()

def split_bounding_box(bb_o, bb_c, axis, division):
    assert 0 <= axis
    assert axis <= 2
    L_o = bb_o
    R_o = torch.cat([bb_o[:axis], division.reshape(1, 1), bb_o[axis + 1:]])

    L_c = torch.cat([bb_c[:axis], division.reshape(1, 1), bb_c[axis + 1:]])
    R_c = bb_c

    return L_o, L_c, R_o, R_c

def build_tree(o, c, V, F):
    sa = bounding_box_area(o, c)

    leaf_cost = C_ISECT * len(V)

    best_cost = maxf
    best_division = None
    best_axis = None
    best_index = None

    for axis in range(3):
        V.sort(key=lambda v: v[axis][0])
        for i, v in enumerate(V):
            division = v[axis]
            L_o, L_c, R_o, R_c = split_bounding_box(o, c, axis, division)
            p_L = bounding_box_area(L_o, L_c) / sa
            p_R = bounding_box_area(R_o, R_c) / sa
            cost = C_TRAV + C_ISECT * (p_L * i + p_R * (len(V) - i))
            if cost < best_cost:
                best_cost = cost
                best_division = division
                best_axis = axis
                best_index = i

    if best_cost > leaf_cost or len(V) <= 1:
        return leaf_cost

    V.sort(key=lambda v: v[best_axis][0])
    L_o, L_c, R_o, R_c = split_bounding_box(o, c, best_axis, best_division)
    cost_L = build_tree(L_o, L_c, V[:best_index], F)
    cost_R = build_tree(R_o, R_c, V[best_index:], F)
    p_L = bounding_box_area(L_o, L_c) / sa
    p_R = bounding_box_area(R_o, R_c) / sa
    return C_TRAV + C_ISECT * (p_L * cost_L + p_R * cost_R)

with open('hack.pbrt') as f:
    template = f.read()

import random

i = 0
while True:
    i += 1

    print(R)
    V = [rotate_vertex(v, R) for v in vertices]
    F = []#[rotate_face(f, R) for f in faces]

    bounding_box_origin = torch.tensor([[maxf], [maxf], [maxf]])
    bounding_box_corner = torch.tensor([[minf], [minf], [minf]])
    for v in V:
        bounding_box_origin = torch.min(bounding_box_origin, v)
        bounding_box_corner = torch.max(bounding_box_corner, v)

    cost = build_tree(bounding_box_origin, bounding_box_corner, V, F)
    print(cost)

    cost.backward()
    R = R - R.grad * 0.001
    R = R.detach()
    R.requires_grad_()

    import os
    with open('out/out-%05d.pbrt' % i, 'w') as f:
        import math
        x = template.replace('ROTATION', '%f %f %f %f' % ( vector_size(R) * 360 / math.pi, R[0].item(), R[1].item(), R[2].item() ))
        x = x.replace('INDEX', '%05d' % i)
        f.write(x)
