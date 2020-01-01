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

C_TRAV = 1 / 80 # in PBRT, apparently
minf = float('-inf')
maxf = float('+inf')








def bounding_box_area(o, c):
    extent = c - o
    return extent.sum() * extent.sum() - (extent * extent).sum()

def split_bounding_box(bb_o, bb_c, axis, division):
    L_o = bb_o
    R_o = torch.cat([bb_o[:axis], division.reshape(1, 1), bb_o[axis + 1:]])

    L_c = torch.cat([bb_c[:axis], division.reshape(1, 1), bb_c[axis + 1:]])
    R_c = bb_c

    return L_o, L_c, R_o, R_c

def cost_of_subdivision(V, F, bb_o, bb_c, axis, division):
    L_tris = []
    R_tris = []
    for f in F:
        if any([f[i][axis].item() < division.item() for i in range(3)]):
            L_tris.append(f)
        if any([f[i][axis].item() > division.item() for i in range(3)]):
            R_tris.append(f)

    L_o, L_c, R_o, R_c = split_bounding_box(bb_o, bb_c, axis, division)
    L_area = bounding_box_area(L_o, L_c)
    R_area = bounding_box_area(R_o, R_c)

    cost = C_TRAV + L_area * len(L_tris) + R_area * len(R_tris)
    return cost, (L_o, L_c, L_tris), (R_o, R_c, R_tris)

def cost_of_cell(V, F, bb_o, bb_c, axis):
    new_axis = (axis + 1) % 1

    my_cost = bounding_box_area(bb_o, bb_c) * len(F)

    best_cost = torch.tensor(maxf)
    best_L = None
    best_R = None
    best_idx = None

#   V.sort(key=lambda v: v[axis].item())
#   F.sort(key=lambda f: min([f[i][axis].item() for i in range(3)]))
    for i, v in enumerate(V):
        division = v[axis]
        cost, L, R = cost_of_subdivision(V, F, bb_o, bb_c, axis, division)
        if cost.item() < best_cost.item():
            best_cost = cost
            best_L = L
            best_R = R
            best_idx = i
    
    if best_cost < my_cost:
        #L_optimized = cost_of_cell(V, L[2], L[0], L[1], new_axis)
        #R_optimized = cost_of_cell(V, R[2], R[0], R[1], new_axis)
        #return C_TRAV + L_optimized + R_optimized
        return best_cost
    else:
        return my_cost


import random
faces = random.sample(faces, 100)

with open('hack.pbrt') as f:
    template = f.read()

i = 0
while True:
    i += 1
    V = [rotate_vertex(v, R) for v in vertices]
    F = [rotate_face(f, R) for f in faces]

    bounding_box_origin = torch.tensor([[maxf], [maxf], [maxf]])
    bounding_box_corner = torch.tensor([[minf], [minf], [minf]])
    for v in V:
        bounding_box_origin = torch.min(bounding_box_origin, v)
        bounding_box_corner = torch.max(bounding_box_corner, v)

    cost = cost_of_cell(V, F, bounding_box_origin, bounding_box_corner, 0)
    print(cost)
    print(R)
    cost.backward()
    R = R - R.grad * 0.01
    R = R.detach()
    R.requires_grad_()

    with open('out/out-%05d.pbrt' % i, 'w') as f:
        import math
        x = template.replace('ROTATION', '%f %f %f %f' % ( vector_size(R) * 360 / math.pi, R[0].item(), R[1].item(), R[2].item() ))
        x = x.replace('INDEX', '%05d' % i)
        f.write(x)
