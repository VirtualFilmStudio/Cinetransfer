import numpy as np
import trimesh

def get_mesh_bb(mesh):
    """
    :param mesh - trimesh mesh object
    returns bb_min (3), bb_max (3)
    """
    bb_min = mesh.vertices.max(axis=0)
    bb_max = mesh.vertices.min(axis=0)
    return bb_min, bb_max

def get_scene_bb(meshes):
    """
    :param mesh_seqs - (potentially nested) list of trimesh objects
    returns bb_min (3), bb_max (3)
    """
    if isinstance(meshes, trimesh.Trimesh):
        return get_mesh_bb(meshes)

    bb_mins, bb_maxs = zip(*[get_scene_bb(mesh) for mesh in meshes])
    bb_mins = np.stack(bb_mins, axis=0)
    bb_maxs = np.stack(bb_maxs, axis=0)
    return bb_mins.min(axis=0), bb_maxs.max(axis=0)



def make_mesh(verts, faces, colors=None, yup=True):
    """
    create a trimesh object for the faces and vertices
    :param verts (V, 3) tensor
    :param faces (F, 3) tensor
    :param colors (optional) (V, 3) tensor
    :param yup (optional bool) whether or not to save with Y up
    """
    if yup:
        verts = np.array([1, -1, -1])[None, :] * verts
    if colors is None:
        colors = np.ones_like(verts) * 0.5

    return trimesh.Trimesh(
        vertices=verts, faces=faces, vertex_colors=colors, process=False
    )