import os
import numpy as np
import trimesh
import imageio

import pyrender

from .geometry.camera import lookat_matrix

class Renderer():

    def __init__(self, W, H, alight=None, bg_color=None):
        self.screen_height = H
        self.screen_width = W
        self.scene = pyrender.Scene(ambient_light=alight, bg_color=bg_color)
        self.renderer = pyrender.OffscreenRenderer(W, H)

    def add_camera(self, pose, visiable=False, wireframe=False, intrins=None, color='norm'):
        """
        H: height of the render image;
        W: width of the render image;
        pose: a numpy array with shape (4, 4)
        """
        self.camera = make_pyrender_camera(self.screen_height, self.screen_width, intrins)
        self.camera_node = self.scene.add(self.camera, pose=pose)
        if visiable:
            camera_marker = make_camera_marker(up='y', color=color)
            camera_marker_node = self.scene.add(
            pyrender.Mesh.from_trimesh(camera_marker, smooth=False, wireframe=wireframe) 
            )
            self.scene.set_pose(camera_marker_node, pose)

        # translate, rotation, scale = transform.get_TRS()
        # node = self._gen_node('camera', camrea, translate, rotation, scale)
        # self.scene.add_node(node)

    def add_light(self, light_type, transform, color=np.ones(3), intensity=1, **kwargs):
        if light_type == "spotlight":
            light = pyrender.SpotLight(color=color, intensity=intensity, **kwargs)
        elif light_type == "directlight":
            light = pyrender.DirectionalLight(color=color, intensity=intensity, **kwargs)
        elif light_type == 'pointlight':
            light = pyrender.PointLight(color=color, intensity=intensity, **kwargs)
        # self.scene.add(light)
        translate, rotation, scale = transform.get_TRS()
        node = self._gen_node("light", light, translate, rotation, scale)
        self.scene.add_node(node)

    def add_mesh(self, mesh, transform):
        """
        mesh: trimesh
        """
        translate, rotation, scale = transform.get_TRS()
        mesh = pyrender.Mesh.from_trimesh(mesh)
        # self.scene.add(mesh)
        node = self._gen_node("mesh", mesh, translate, rotation, scale)
        self.scene.add_node(node)
        return node

    def _gen_node(self, node_type, content, tranlate=(0,0,0), rotation=(0,0,0,1), scale=(1,1,1)):
        if node_type == "mesh":
            node = pyrender.Node(mesh=content, translation=tranlate, rotation=rotation,scale=scale)
        elif node_type == 'camera':
            node = pyrender.Node(camera=content, translation=tranlate, rotation=rotation,scale=scale)
        elif node_type == 'light':
            node = pyrender.Node(light=content, translation=tranlate, rotation=rotation,scale=scale)
        return node

    def render(self, flags):
        """
        return:
        color: (h, w, 3)
        depth: (h, w)
        """
        return self.renderer.render(self.scene, flags)
    
class Transform():

    def __init__(self, translate=(0,0,0), rotation=(0,0,0,1), scale=(1,1,1)) -> None:
        self.translate = translate
        self.rotation = rotation
        self.scale = scale

    def get_TRS(self):
        return self.translate, self.rotation, self.scale
    
    def get_matrix(self):
        pass

def make_pyrender_camera(H, W, intrins=None):
    if intrins is not None:
        print("USING INTRINSICS CAMERA", intrins)
        fx, fy, cx, cy = intrins
        return pyrender.IntrinsicsCamera(fx, fy, cx, cy)

    focal = 0.5 * (H + W)
    yfov = 2 * np.arctan(0.5 * H / focal)
    print("USING PERSPECTIVE CAMERA", H, W, focal)
    return pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=W / H)


def make_camera_marker(radius=0.1, height=0.2, up="y", transform=None, color='norm'):
    """
    :param radius (default 0.1) radius of pyramid base, diagonal of image plane
    :param height (default 0.2) height of pyramid, focal length
    :param up (default y) camera up vector
    :param transform (default None) (4, 4) cam to world transform
    """
    verts, faces, face_colors = camera_marker_geometry(radius, height, up)
    if transform is not None:
        assert transform.shape == (4, 4)
        verts = (
            np.einsum("ij,nj->ni", transform[:3, :3], verts) + transform[None, :3, 3]
        )
    if color == 'red':
        face_colors = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        )
    elif color == 'green':
        face_colors = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
            ]
        )
    elif color == 'blue':
        face_colors = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        )
    elif color == 'yellow':
        face_colors = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
            ]
        )
    elif color == 'cyan':
        face_colors = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
            ]
        )
    else:
        face_colors = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        )
    
    return trimesh.Trimesh(verts, faces, face_colors=face_colors, process=False)


def camera_marker_geometry(radius, height, up):
    assert up == "y" or up == "z"
    if up == "y":
        vertices = np.array(
            [
                [-radius, -radius, 0],
                [radius, -radius, 0],
                [radius, radius, 0],
                [-radius, radius, 0],
                [0, 0, height],
            ]
        )
    else:
        vertices = np.array(
            [
                [-radius, 0, -radius],
                [radius, 0, -radius],
                [radius, 0, radius],
                [-radius, 0, radius],
                [0, -height, 0],
            ]
        )

    faces = np.array(
        [[0, 3, 1], [1, 3, 2], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],]
    )

    face_colors = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )
    return vertices, faces, face_colors