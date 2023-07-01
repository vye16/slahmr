import os
import imageio
import numpy as np

import time
import torch
import trimesh

os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyrender
from pyrender.constants import RenderFlags
from pyrender.light import DirectionalLight
from pyrender.node import Node

from slahmr.geometry.camera import make_rotation, make_translation
from .tools import (
    read_image,
    checkerboard_geometry,
    camera_marker_geometry,
    transform_pyrender,
)


def init_viewer(
    img_size, intrins, vis_scale=1.0, bg_paths=None, fps=30,
):
    img_size = int(vis_scale * img_size[0]), int(vis_scale * img_size[1])
    intrins = vis_scale * intrins

    platform = os.environ.get("PYOPENGL_PLATFORM", "pyglet")
    if platform == "pyglet":
        vis = AnimationViewer(img_size, intrins=intrins, fps=fps)
        print("VIS", vis)
        return vis

    if platform != "egl":
        raise NotImplementedError

    vis = OffscreenAnimation(img_size, intrins=intrins, fps=fps)
    if bg_paths is not None:
        bg_imgs = [
            read_image(p, scale=vis_scale).astype(np.float32) / 255 for p in bg_paths
        ]
        vis.set_bg_seq(bg_imgs)
    print("VIS", vis)
    return vis


class AnimationBase(object):
    """
    Render an animation
    IMPORTANT: pyrender uses coordinates with Y UP (i.e. xyz = right up back),
    we must keep this in mind because the human prior frame has Z UP (i.e.
    xyz = right forward up)
    """

    def __init__(self, img_size, intrins=None):
        self.scene = pyrender.Scene(
            ambient_light=[0.3, 0.3, 0.3], bg_color=[1.0, 1.0, 1.0, 0.0]
        )
        self.scene.bg_color = np.array([1.0, 1.0, 1.0, 0.0])

        self.viewport_size = img_size
        self.camera = make_pyrender_camera(img_size, intrins)

        # set the ground with Z up
        self.ground_pose = np.eye(4)
        ground = pyrender.Mesh.from_trimesh(
            make_checkerboard(up="y", alpha=1.0), smooth=False
        )
        self.ground_node = self.scene.add(ground, name="ground", pose=self.ground_pose)
        self.cam_node = self.scene.add(self.camera, name="camera")

        self.cam_marker_node = None
        self.cam_marker_poses = []

        self.light_nodes = []
        self.light_poses = []

        self.bg_seq = []

        # pyrender nodes, reused between frames
        self.anim_nodes = []
        # list of meshes per frame of animation
        self.anim_meshes = []
        # list of camera poses per frame of animation
        self.anim_cameras = None
        # current timestep of animation
        self.anim_idx = 0

    def acquire_lock(self):
        pass

    def release_lock(self):
        pass

    def close(self):
        pass

    def delete(self):
        pass

    def clear_meshes(self):
        self.anim_meshes = []
        self.anim_idx = 0

    @property
    def anim_len(self):
        return len(self.anim_meshes)

    def add_lighting(self, color=np.ones(3), intensity=1.0):
        self.light_poses = get_light_poses()
        self.light_poses.append(np.eye(4))
        cam_pose = self.scene.get_pose(self.cam_node)
        for i, pose in enumerate(self.light_poses):
            matrix = cam_pose @ pose
            node = Node(
                name=f"light-{i:02d}",
                light=DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if self.scene.has_node(node):
                continue
            self.scene.add_node(node)
            self.light_nodes.append(node)

    def add_mesh_frame(self, meshes, debug=False):
        """
        :param meshes (list) trimesh meshes for this frame timestep
        """
        meshes = [pyrender.Mesh.from_trimesh(m) for m in meshes]
        num_nodes = len(meshes)
        self.anim_meshes.append(meshes)

        if num_nodes > len(self.anim_nodes):
            # need to add a node in the pyrender scene
            self.acquire_lock()
            for i in range(len(self.anim_nodes), num_nodes):
                self.anim_nodes.append(self.scene.add(meshes[i], name=f"mesh_{i:03d}"))
            self.release_lock()

    def add_static_meshes(self, meshes, smooth=True):
        """
        add all meshes to a single frame (timestep)
        """
        meshes = [pyrender.Mesh.from_trimesh(m, smooth=smooth) for m in meshes]
        if len(self.anim_meshes) < 1:
            self.anim_meshes.append([])
        self.anim_meshes[0].extend(meshes)
        num_meshes = len(self.anim_meshes[0])
        num_nodes = len(self.anim_nodes)
        if num_meshes > num_nodes:
            # need to add a node in the pyrender scene
            self.acquire_lock()
            self.anim_nodes.extend(
                [
                    self.scene.add(self.anim_meshes[0][i])
                    for i in range(num_nodes, num_meshes)
                ]
            )
        self.release_lock()

    def set_camera_seq(self, poses):
        """
        :param poses (*, 4, 4)
        """
        poses = transform_pyrender(poses)
        poses = np.asarray(poses)
        assert poses.ndim >= 2 and poses.shape[-2:] == (4, 4)
        if poses.ndim < 3:
            poses = poses[None]
        print(f"Adding camera sequence length {len(poses)}")
        self.anim_cameras = poses

    def add_camera_markers(self, poses):
        """
        Add a single camera marker node that we move around when we animate
        """
        print("ADDING CAMERA MARKERS")
        if self.cam_marker_node is None:
            cam_marker = make_camera_marker(up="y")
            self.cam_marker_node = self.scene.add(
                pyrender.Mesh.from_trimesh(cam_marker, smooth=False)
            )
        poses = transform_pyrender(poses)
        poses = np.asarray(poses)
        self.cam_marker_poses = poses

    def add_camera_markers_static(self, poses):
        """
        Add all camera markers as separate mesh nodes to render in one pass
        """
        print("ADDING STATIC CAMERA MARKERS")
        poses = transform_pyrender(poses)
        poses = np.asarray(poses)
        cam_meshes = [make_camera_marker(up="y", transform=pose) for pose in poses]
        self.add_static_meshes(cam_meshes, smooth=False)

    def update_camera(self, cam_pose):
        self.acquire_lock()
        self.scene.set_pose(self.cam_node, pose=cam_pose)
        self.release_lock()

    def set_ground(self, pose):
        pose = transform_pyrender(pose)
        print("Setting ground pose to be", pose)
        self.ground_pose = np.asarray(pose)
        self.scene.set_pose(self.ground_node, pose=self.ground_pose)

    def set_bg_seq(self, bg_imgs):
        pass

    def set_mesh_visibility(self, vis):
        self.acquire_lock()
        for node in self.anim_nodes:
            node.mesh.is_visible = vis
        self.release_lock()

    def check_mesh_visibility(self):
        for node in self.anim_nodes:
            if node.mesh.is_visible:
                return True

        if self.ground_node.mesh.is_visible:
            return True

        if self.cam_marker_node is not None and self.cam_marker_node.mesh.is_visible:
            return True

        return False

    def update_frame(self):
        t = self.anim_idx % self.anim_len

        # update camera
        if self.anim_cameras is not None:
            cam_t = min(t, len(self.anim_cameras) - 1)
            cam_pose = self.anim_cameras[cam_t]
            self.update_camera(cam_pose)
            if self.cam_marker_node is not None and t < len(self.cam_marker_poses):
                self.acquire_lock()
                self.scene.set_pose(self.cam_marker_node, self.cam_marker_poses[t])
                self.release_lock()

        # update meshes
        meshes = self.anim_meshes[t]
        self.acquire_lock()
        for i, node in enumerate(self.anim_nodes):
            if i < len(meshes):
                node.mesh = meshes[i]
                node.mesh.is_visible = True
            else:
                node.mesh.is_visible = False
        self.release_lock()


class OffscreenAnimation(AnimationBase):
    def __init__(self, img_size, intrins=None, fps=30, ext="mp4"):
        super().__init__(img_size, intrins=intrins)
        self.add_lighting(0.9)

        self.fps = fps
        self.ext = ext
        self.viewer = pyrender.OffscreenRenderer(*self.viewport_size)

        self.bg_seq = []

    def set_bg_seq(self, bg_imgs):
        assert isinstance(bg_imgs, list)
        print(f"setting length {len(bg_imgs)} bg sequence")
        self.bg_seq = bg_imgs

    def delete(self):
        self.viewer.delete()

    def close(self):
        self.delete()

    def render(
        self, render_bg=True, render_ground=True, render_cam=True, fac=1.0, **kwargs
    ):
        flags = RenderFlags.RGBA | RenderFlags.SHADOWS_DIRECTIONAL
        self.ground_node.mesh.is_visible = render_ground and (not render_bg)
        if self.cam_marker_node is not None:
            self.cam_marker_node.mesh.is_visible = render_cam

        rgba = np.zeros((*self.viewport_size[::-1], 4))
        if self.check_mesh_visibility():  # if any meshes are visible
            rgba, _ = self.viewer.render(self.scene, flags=flags)

        img = rgba
        if render_bg and len(self.bg_seq) > 0:
            t = min(self.anim_idx, len(self.bg_seq) - 1)
            bg = self.bg_seq[t]
            rgba = rgba.astype(np.float32) / 255
            alpha = fac * rgba[..., 3:]
            img = alpha * rgba[..., :3] + (1 - alpha) * bg
            img = (255 * img).astype(np.uint8)

        return img

    def render_mesh_layers(self, render_ground=True, render_cam=True, **kwargs):
        # render each mesh in its own layer
        flags = RenderFlags.RGBA | RenderFlags.SHADOWS_DIRECTIONAL
        render_cam = render_cam and self.cam_marker_node is not None

        # current timestep
        t = self.anim_idx % self.anim_len
        meshes = self.anim_meshes[t]
        layers = []

        # toggle ground and camera
        self.ground_node.mesh.is_visible = False
        if self.cam_marker_node is not None:
            self.cam_marker_node.mesh.is_visible = False

        # render animation meshes for current timestep
        for i in range(len(meshes)):
            for j, node in enumerate(self.anim_nodes):
                if i == j:
                    node.mesh.is_visible = True
                else:
                    node.mesh.is_visible = False
            rgba, _ = self.viewer.render(self.scene, flags=flags)
            layers.append(rgba)

        if not render_ground and not render_cam:
            return layers

        # render the ground and camera
        for node in self.anim_nodes:
            node.mesh.is_visible = False

        if render_cam:
            self.cam_marker_node.mesh.is_visible = render_cam
            rgba, _ = self.viewer.render(self.scene, flags=flags)
            layers.append(rgba)
            self.cam_marker_node.mesh.is_visible = False

        if render_ground:
            self.ground_node.mesh.is_visible = render_ground
            rgba, _ = self.viewer.render(self.scene, flags=flags)
            layers.append(rgba)

        return layers

    def render_frames_layers(self):
        frames_layers = []
        for t in range(self.anim_len):
            self.anim_idx = t
            self.update_frame()
            layers = self.render_mesh_layers()
            frames_layers.append(layers)
        return frames_layers

    def animate(self, save_name, save_frames=False, render_layers=False, **kwargs):
        if render_layers:
            frames_layers = self.render_frames_layers()
            os.makedirs(save_name, exist_ok=True)
            for t, layers in enumerate(frames_layers):
                for l, layer in enumerate(layers):
                    path = f"{save_name}/{t:06d}_{l:03d}.png"
                    imageio.imwrite(path, layer)
            print(f"saved frame layers to {save_name}")
            return save_name

        frames = self.render_frames(**kwargs)
        if len(frames) > 1:
            if save_frames:
                os.makedirs(save_name, exist_ok=True)
                for i, frame in enumerate(frames):
                    path = f"{save_name}/{i:06d}.png"
                    imageio.imwrite(path, frame)
                return save_name
            save_path = f"{save_name}.{self.ext}"
            imageio.mimwrite(save_path, frames, fps=self.fps)
            print("wrote video to", save_path)
            return save_path

        save_path = f"{save_name}.png"
        imageio.imwrite(save_path, frames[0])
        print("wrote image to", save_path)
        return save_path

    def render_frames(self, **kwargs):
        print("ANIMATION LENGTH", self.anim_len)
        frames = []
        for t in range(self.anim_len):
            self.anim_idx = t
            self.update_frame()
            img = self.render(**kwargs)
            frames.append(img)
        return frames

    def render_layers(self, save_dir, composite=False, fac=0.4):
        os.makedirs(save_dir, exist_ok=True)

        # render all frames
        frames = self.render_frames(render_bg=False, render_ground=False)

        # render background
        self.set_mesh_visibility(False)
        bg_img = self.render(render_bg=False, render_ground=True) / 255

        # save frames
        for t, frame in enumerate(frames):
            imageio.imwrite(f"{save_dir}/{t:06d}.png", frame)
        imageio.imwrite(f"{save_dir}/background.png", bg_img)

        if composite:
            rgba = composite_layers(frames, bg_img, fac=fac)
            imageio.imwrite(f"{save_dir}/composite.png", (255 * rgba).astype(np.uint8))


def composite_layers(frames, bg_img, fac=0.4):
    # composite front to back
    H, W = frames[0].shape[:2]
    comp = np.zeros((H, W, 3))
    vis = np.ones((H, W, 1))
    for frame in frames:
        frame = frame / 255
        alpha = fac * frame[..., 3:]
        comp += vis * alpha * frame[..., :3]
        vis *= 1 - alpha

    comp += vis * bg_img[..., 3:] * bg_img[..., :3]
    vis *= 1 - bg_img[..., 3:]
    return comp + vis
    rgba = np.concatenate([comp, 1 - vis], axis=-1)
    return rgba


class AnimationViewer(AnimationBase):
    def __init__(self, img_size, intrins=None, fps=30, num_repeats=1):
        super().__init__(img_size, intrins=intrins)

        self.fps = fps
        self.ext = "gif"

        def pause_play_callback(_, viewer):
            viewer.is_paused = not viewer.is_paused

        def step_callback(_, viewer, step):
            viewer.anim_idx = (viewer.anim_idx + step) % viewer.anim_len

        def stop_callback(_, viewer):
            viewer.do_animate = False

        def loop_callback(_, viewer):
            viewer.num_repeats *= -1

        registered_keys = {
            "p": (pause_play_callback, [self]),
            ".": (step_callback, [self, 1]),
            ",": (step_callback, [self, -1]),
            "l": (loop_callback, [self]),
            "s": (stop_callback, [self]),
        }

        self.viewer = pyrender.Viewer(
            self.scene,
            run_in_thread=True,
            viewport_size=self.viewport_size,
            use_raymond_lighting=True,
            registered_keys=registered_keys,
        )

        self.num_repeats = num_repeats
        self.do_animate = False
        self.is_paused = False

    def acquire_lock(self):
        self.viewer.render_lock.acquire()

    def release_lock(self):
        self.viewer.render_lock.release()

    def close(self):
        self.viewer.close_external()

    def animate(self, save_name=None, **kwargs):
        print("===================")
        print(f"PLAYING ANIMATION {self.num_repeats} TIMES")
        print("VIEWER CONTROLS:")
        print("p\tpause/play")
        print("l\tloop/unloop")
        print(".\tstep forward")
        print(",\tstep backward")
        print("s\tstop")
        print("===================")

        self.anim_idx = 0
        self.do_animate = True
        frame_dur = 1 / self.fps
        if save_name is not None:
            self.viewer.viewer_flags["record"] = True

        # set up initial frame
        self.update_frame()

        while self.do_animate:
            if self.is_paused:
                self.update_frame()
                continue

            if (
                self.num_repeats > 0
                and self.anim_idx >= self.num_repeats * self.anim_len
            ):
                break

            if save_name is not None and self.anim_idx >= self.anim_len:
                self.viewer.viewer_flags["record"] = False

            start_time = time.time()
            self.update_frame()
            render_time = time.time() - start_time

            self.anim_idx += 1
            sleep_time = frame_dur - render_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        if save_name is not None:
            print(f"SAVING RECORDING TO {save_name}")
            self.viewer.save_gif(f"{save_name}.{self.ext}")


def make_checkerboard(
    length=25.0,
    color0=[0.8, 0.9, 0.9],
    color1=[0.6, 0.7, 0.7],
    tile_width=0.5,
    alpha=1.0,
    up="y",
):
    v, f, _, fc = checkerboard_geometry(length, color0, color1, tile_width, alpha, up)
    return trimesh.Trimesh(v, f, face_colors=fc, process=False)


def make_pyrender_camera(img_size, intrins=None):
    if intrins is not None:
        print("USING INTRINSICS CAMERA", intrins)
        fx, fy, cx, cy = intrins
        return pyrender.IntrinsicsCamera(fx, fy, cx, cy)

    W, H = img_size
    focal = 0.5 * (H + W)
    yfov = 2 * np.arctan(0.5 * H / focal)
    print("USING PERSPECTIVE CAMERA", H, W, focal)
    return pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=W / H)


def make_camera_marker(radius=0.1, height=0.2, up="y", transform=None):
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
    return trimesh.Trimesh(verts, faces, face_colors=face_colors, process=False)


def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses


def get_raymond_light_poses(up="z"):
    thetas = np.pi * np.ones(3) / 6
    phis = 2 * np.pi * np.arange(3) / 3

    poses = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        if up == "y":
            z = np.array([xp, zp, -yp])
        else:
            z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        poses.append(matrix)
    return poses
