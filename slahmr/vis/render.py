import torch
import torch.nn as nn
import numpy as np

from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights,
    PerspectiveCameras,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer import TexturesAtlas, TexturesVertex
from pytorch3d.transforms import euler_angles_to_matrix

from .tools import get_colors, read_image, transform_torch3d, checkerboard_geometry


def init_renderer(img_size, intrins, device, vis_scale=1.0, bg_paths=None):
    img_size = int(vis_scale * img_size[0]), int(vis_scale * img_size[1])
    renderer = RenderBase(device, img_size, intrins=vis_scale * intrins)
    if bg_paths is not None:
        bg_imgs = [
            torch.from_numpy(read_image(p, scale=vis_scale)).float() / 255
            for p in bg_paths
        ]
        print(bg_imgs[0].shape)
        renderer.set_bg_seq(bg_imgs)
    print("RENDERER", renderer)
    return renderer


class RenderBase(nn.Module):
    def __init__(
        self,
        device,
        img_size,
        intrins=None,
        blur_radius: float = 0.0,  # rasterizer settings
        faces_per_pixel: int = 3,
    ):
        """
        Uses Pytorch3D to render results.
        We use the PyTorch3D coordinate system, which assumes:
        +X:left, +Y: up and +Z: from us to scene (right-handed)
        https://pytorch3d.org/docs/cameras
        """
        super().__init__()
        self.device = device
        W, H = img_size
        self.img_size = (H, W)
        if intrins is not None:
            intrins = intrins.to(device)
            self.cam_f, self.cam_center = intrins[:2], intrins[2:]
            cx, cy = self.cam_center.detach().cpu()
        else:
            self.cam_f = torch.tensor([0.5 * (H + W), 0.5 * (H + W)], device=device)
            self.cam_center = torch.tensor([0.5 * W, 0.5 * H], device=device)
        print("cam_f", self.cam_f, "cam_center", self.cam_center)
        print("img_size", self.img_size)
        self.blur_radius = blur_radius
        self.faces_per_pixel = faces_per_pixel

        self.light_loc = [[0, 0, -0.5]]
        self.cameras = self.make_cameras()
        self.bg_imgs = None
        self.set_ground()
        self.build_renderer()

    def build_renderer(self):
        self.lights = PointLights(device=self.device, location=self.light_loc)

        self.rasterizer = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=self.blur_radius,
            faces_per_pixel=self.faces_per_pixel,
            max_faces_per_bin=100000,
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, raster_settings=self.rasterizer
            ),
            shader=HardPhongShader(
                cameras=self.cameras, lights=self.lights, device=self.device
            ),
        )

    def make_cameras(self, cam_poses=None):
        device = self.device
        if cam_poses is None:
            cam_poses = torch.eye(4, device=device)[None]
        cam_poses = cam_poses.to(device)
        cam_R, cam_t = transform_torch3d(cam_poses)
        Nc = len(cam_poses)
        focal = self.cam_f[None].repeat(Nc, 1)
        ppt = self.cam_center[None].repeat(Nc, 1)
        img_size = [self.img_size for _ in range(Nc)]
        return PerspectiveCameras(
            device=self.device,
            in_ndc=False,
            focal_length=focal,
            principal_point=ppt,
            image_size=img_size,
            R=cam_R,
            T=cam_t,
        )

    def render_frame(self, cam_poses, verts, faces, colors):
        """
        :param cam_poses (C, 4, 4)
        :param verts (B, V, 3)
        :param faces (F, 3)
        :param colors (B, 4)
        """
        with torch.no_grad():
            cameras = self.make_cameras(cam_poses)
            mesh = build_meshes(verts, faces, colors)
            image = self.renderer(mesh, cameras=cameras)
        return image

    def render_with_ground(self, cam_poses, verts, faces, colors, ground_pose):
        """
        :param cam_poses (C, 4, 4)
        :param verts (B, V, 3)
        :param faces (F, 3)
        :param colors (B, 3)
        :param ground_pose (4, 4)
        """
        with torch.no_grad():
            self.set_ground_pose(ground_pose)
            cameras = self.make_cameras(cam_poses)
            # (B, V, 3), (B, F, 3), (B, V, 3)
            verts, faces, colors = prep_shared_geometry(verts, faces, colors)
            # (V, 3), (F, 3), (V, 3)
            gv, gf, gc = self.ground_geometry
            verts = list(torch.unbind(verts, dim=0)) + [gv]
            faces = list(torch.unbind(faces, dim=0)) + [gf]
            colors = list(torch.unbind(colors, dim=0)) + [gc[..., :3]]
            mesh = create_meshes(verts, faces, colors)

            image = self.renderer(mesh, cameras=cameras)
        return image

    def set_bg_seq(self, bg_imgs):
        self.bg_imgs = bg_imgs

    def set_ground(self, pose=None):
        device = self.device
        v, f, vc, fc = map(torch.from_numpy, checkerboard_geometry(up="y"))
        v, f, vc = v.to(device), f.to(device), vc.to(device)
        self.ground_geometry = [v, f, vc]
        if pose is not None:
            self.set_ground_pose(pose)

    def set_ground_pose(self, pose):
        """
        :param pose (4, 4)
        """
        if self.ground_geometry is None:
            return
        v = self.ground_geometry[0]
        pose = pose.to(self.device)
        #         R, t = pose[:3, :3], pose[:3, 3]
        R, t = transform_torch3d(pose)
        self.ground_geometry[0] = torch.einsum("ij,vj->vi", R, v) + t[None]

    def composite_bg(self, frames, fac=1.0):
        T = len(frames)
        if self.bg_imgs is None:
            return frames

        out_frames = []
        for bg, frame in zip(self.bg_imgs, frames):
            alpha = fac * frame[..., 3:]
            out = alpha * frame[..., :3] + (1 - alpha) * bg
            out_frames.append(out)
        return out_frames

    def render_video(
        self,
        cam_poses,
        verts,
        faces,
        colors,
        render_bg=False,
        fac=1.0,
        ground_pose=None,
    ):
        """
        :param cam_poses (C, T, 4, 4)
        :param verts list of T (B, V, 3)
        :param faces list of T (F, 3)
        :param colors list of T (B, 4)
        :param render_bg (optional default False)
        :param fac (optional float, default 1)
        :param ground_pose (optional, default None) (4, 4)
        returns list of T (H, W, 4)
        """
        T = len(verts)
        print(f"rendering video with {T} frames")
        if cam_poses.shape[1] == 1:
            cam_poses = cam_poses.expand(-1, T, -1, -1)

        if render_bg:  # don't render ground on source frame
            ground_pose = None

        frames = []
        for t in range(T):
            if ground_pose is None:
                # (H, W, 4)
                frame = self.render_frame(
                    cam_poses[:, t], verts[t], faces[t], colors[t]
                )
            else:
                # (H, W, 4)
                frame = self.render_with_ground(
                    cam_poses[:, t], verts[t], faces[t], colors[t], ground_pose
                )
            frames.append(frame.cpu()[0])

        if render_bg and self.bg_imgs is not None:
            frames = self.composite_bg(frames, fac=fac)

        return [(255 * f).byte() for f in frames]


def build_meshes(verts, faces, colors):
    return create_meshes(*prep_shared_geometry(verts, faces, colors))


def prep_shared_geometry(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (F, 3)
    :param colors (B, 4)
    """
    B, V, _ = verts.shape
    F, _ = faces.shape
    colors = colors.unsqueeze(1).expand(B, V, -1)[..., :3]
    faces = faces.unsqueeze(0).expand(B, F, -1)
    return verts, faces, colors


def create_meshes(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (B, F, 3)
    :param colors (B, V, 3)
    """
    textures = TexturesVertex(verts_features=colors)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)
    return join_meshes_as_scene(meshes)
