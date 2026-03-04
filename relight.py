"""Relight nvdiffrecmc-optimized scenes with new environment maps.

Given:
  - An optimized scene mesh (from nvdiffrecmc training output)
  - Relight metadata specifying which lighting to swap
  - Per-view environment maps stored as HDR/LDR PNG pairs

This script reconstructs approximate HDR environment maps from the stored
PNG pairs (undoing the log1p transform from preprocess_objaverse.py),
replaces the optimized scene's probe, and renders all target views.

Usage:
  # Single scene
  python relight.py \
    --relight-meta data_samples/relight_metadata/Barrel_02_env_2.json \
    --scene-meta-root /path/to/test/metadata \
    --envmaps-root /path/to/test/envmaps \
    --mesh-root out/polyhaven \
    --output-dir relight_output

  # Batch: process all relight metadata files
  python relight.py \
    --relight-meta-dir data_samples/relight_metadata \
    --scene-meta-root /path/to/test/metadata \
    --envmaps-root /path/to/test/envmaps \
    --mesh-root out/polyhaven \
    --output-dir relight_output
"""

import os
import sys
import json
import argparse
import glob

import numpy as np
import torch
import nvdiffrast.torch as dr
import imageio

from render import mesh, light, render, util, obj
import render.optixutils as ou


OPENCV_TO_OPENGL = torch.tensor([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
], dtype=torch.float32)


def reconstruct_hdr_from_pngs(hdr_path, ldr_path):
    """Reconstruct approximate HDR from the stored hdr+ldr PNG pair.

    The preprocessing (preprocess_objaverse.py) saves:
      hdr_png = uint8(clip(log1p(10 * raw) / max_log, 0, 1) * 255)
      ldr_png = uint8(clip(raw, 0, 1)^(1/2.2) * 255)

    We invert this by estimating max_log from non-saturated LDR pixels,
    then recovering raw = expm1(hdr_norm * max_log) / 10.
    """
    hdr_img = imageio.imread(hdr_path).astype(np.float32) / 255.0
    ldr_img = imageio.imread(ldr_path).astype(np.float32) / 255.0

    if hdr_img.ndim == 2:
        hdr_img = hdr_img[..., None].repeat(3, axis=-1)
    if ldr_img.ndim == 2:
        ldr_img = ldr_img[..., None].repeat(3, axis=-1)
    hdr_img = hdr_img[..., :3]
    ldr_img = ldr_img[..., :3]

    raw_from_ldr = np.power(ldr_img, 2.2)

    # Estimate max_log from non-saturated pixels
    not_saturated = raw_from_ldr.max(axis=-1) < 0.95
    has_signal = hdr_img.max(axis=-1) > 0.02
    mask = not_saturated & has_signal

    if mask.sum() > 100:
        ratio = np.log1p(10.0 * raw_from_ldr[mask]) / np.clip(hdr_img[mask], 1e-6, None)
        max_log = float(np.median(ratio))
    else:
        max_log = float(np.log1p(10.0))

    max_log = max(max_log, float(np.log1p(10.0)))

    raw_reconstructed = np.expm1(hdr_img * max_log) / 10.0
    return np.clip(raw_reconstructed, 0, None).astype(np.float32)


def load_envmap_for_scene(envmaps_dir, scene_name, frame_idx=0):
    """Load and reconstruct an HDR environment map from the per-view PNGs.

    Uses a single frame's HDR/LDR pair to reconstruct the approximate HDR.
    """
    scene_envmap_dir = os.path.join(envmaps_dir, scene_name)
    if not os.path.isdir(scene_envmap_dir):
        raise FileNotFoundError(f"Envmap directory not found: {scene_envmap_dir}")

    hdr_path = os.path.join(scene_envmap_dir, f"{frame_idx:05d}_hdr.png")
    ldr_path = os.path.join(scene_envmap_dir, f"{frame_idx:05d}_ldr.png")

    if not os.path.isfile(hdr_path) or not os.path.isfile(ldr_path):
        available = sorted(glob.glob(os.path.join(scene_envmap_dir, "*_hdr.png")))
        if not available:
            raise FileNotFoundError(f"No HDR envmap PNGs in {scene_envmap_dir}")
        hdr_path = available[0]
        ldr_path = hdr_path.replace("_hdr.png", "_ldr.png")
        if not os.path.isfile(ldr_path):
            raise FileNotFoundError(f"LDR counterpart missing: {ldr_path}")

    raw_hdr = reconstruct_hdr_from_pngs(hdr_path, ldr_path)
    return raw_hdr


def build_camera(frame, resolution, cam_near_far):
    """Build mv, mvp, campos from a metadata frame entry."""
    fx, fy, cx, cy = frame['fxfycxcy']
    H = resolution[0]
    aspect = resolution[1] / resolution[0]
    fovy = 2.0 * np.arctan(0.5 * H / fy)

    proj = util.perspective(fovy, aspect, cam_near_far[0], cam_near_far[1])
    w2c = torch.tensor(frame['w2c'], dtype=torch.float32)
    mv = OPENCV_TO_OPENGL @ w2c
    campos = torch.linalg.inv(mv)[:3, 3]
    mvp = proj @ mv

    return mv, mvp, campos


def make_flags(args):
    """Create a minimal FLAGS namespace for render_mesh."""
    flags = argparse.Namespace()
    flags.n_samples = args.n_samples
    flags.bsdf = 'pbr'
    flags.decorrelated = False
    flags.denoiser = 'none'
    flags.denoiser_demodulate = True
    flags.layers = 1
    flags.spp = args.spp
    flags.train_res = [args.res, args.res]
    flags.display_res = [args.res, args.res]
    flags.background = 'black'
    flags.cam_near_far = [0.1, 1000.0]
    flags.transparency = False
    flags.no_perturbed_nrm = False
    return flags


def relight_scene(relight_meta_path, scene_meta_root, envmaps_root, mesh_root,
                  output_dir, args, glctx):
    """Relight a single scene according to its relight metadata."""

    with open(relight_meta_path) as f:
        relight_meta = json.load(f)

    scene_name = relight_meta['scene_name']
    relit_scene_name = relight_meta['relit_scene_name']
    target_indices = relight_meta['target_view_indices']

    # Load scene metadata (for camera poses)
    scene_meta_path = os.path.join(scene_meta_root, f"{scene_name}.json")
    if not os.path.isfile(scene_meta_path):
        print(f"  [SKIP] Scene metadata not found: {scene_meta_path}")
        return False

    with open(scene_meta_path) as f:
        scene_meta = json.load(f)

    all_frames = scene_meta['frames']

    # Load optimized mesh
    mesh_dir = os.path.join(mesh_root, scene_name, "mesh")
    mesh_obj_path = os.path.join(mesh_dir, "mesh.obj")
    if not os.path.isfile(mesh_obj_path):
        print(f"  [SKIP] Optimized mesh not found: {mesh_obj_path}")
        return False

    loaded_mesh = mesh.load_mesh(mesh_obj_path)
    loaded_mesh = mesh.auto_normals(loaded_mesh)
    loaded_mesh = mesh.compute_tangents(loaded_mesh)

    # Build OptiX BVH for ray tracing
    optix_ctx = ou.OptiXContext()
    ou.optix_build_bvh(optix_ctx, loaded_mesh.v_pos.contiguous(),
                       loaded_mesh.t_pos_idx.int(), rebuild=1)

    # Reconstruct HDR envmap from the relit scene
    print(f"  Loading envmap from '{relit_scene_name}'...")
    raw_hdr = load_envmap_for_scene(envmaps_root, relit_scene_name)
    env_tensor = torch.tensor(raw_hdr, dtype=torch.float32, device='cuda')

    # Optionally resize envmap to probe_res
    if args.probe_res is not None:
        texcoord = util.pixel_grid(args.probe_res, args.probe_res)
        env_tensor = torch.clamp(
            dr.texture(env_tensor[None, ...], texcoord[None, ...],
                       filter_mode='linear')[0],
            min=0.0001)

    lgt = light.EnvironmentLight(base=env_tensor)

    FLAGS = make_flags(args)
    resolution = FLAGS.train_res

    # Determine image resolution from first available target frame
    sample_frame = all_frames[target_indices[0]]
    sample_img_path = sample_frame['image_path']
    if os.path.isfile(sample_img_path):
        sample_img = imageio.imread(sample_img_path)
        resolution = [sample_img.shape[0], sample_img.shape[1]]
        FLAGS.train_res = resolution
        FLAGS.display_res = resolution

    out_scene_dir = os.path.join(output_dir, f"{scene_name}_relit_by_{relit_scene_name}")
    os.makedirs(out_scene_dir, exist_ok=True)

    print(f"  Rendering {len(target_indices)} views at {resolution}...")
    for view_idx in target_indices:
        if view_idx >= len(all_frames):
            print(f"  [WARN] View index {view_idx} out of range ({len(all_frames)} frames), skipping")
            continue

        frame = all_frames[view_idx]
        mv, mvp, campos = build_camera(frame, resolution, FLAGS.cam_near_far)

        mvp_cuda = mvp[None, ...].cuda()
        campos_cuda = campos[None, ...].cuda()

        with torch.no_grad():
            buffers = render.render_mesh(
                FLAGS, glctx, loaded_mesh,
                mvp_cuda, campos_cuda, lgt, resolution,
                spp=FLAGS.spp, num_layers=FLAGS.layers,
                background=None, optix_ctx=optix_ctx)

        shaded = buffers['shaded'][0, ..., 0:3]
        shaded_srgb = util.rgb_to_srgb(shaded)
        shaded_np = torch.clamp(shaded_srgb, 0.0, 1.0).detach().cpu().numpy()

        out_path = os.path.join(out_scene_dir, f"view_{view_idx:05d}.png")
        util.save_image(out_path, shaded_np)

    # Save the reconstructed probe for reference
    light.save_env_map(os.path.join(out_scene_dir, "probe.hdr"), lgt)

    # Save a metadata summary
    summary = {
        "scene_name": scene_name,
        "relit_scene_name": relit_scene_name,
        "target_view_indices": target_indices,
        "resolution": resolution,
        "n_samples": FLAGS.n_samples,
    }
    with open(os.path.join(out_scene_dir, "relight_info.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Done -> {out_scene_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Relight nvdiffrecmc scenes')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--relight-meta', type=str,
                       help='Single relight metadata JSON file')
    group.add_argument('--relight-meta-dir', type=str,
                       help='Directory of relight metadata JSON files (batch mode)')

    parser.add_argument('--scene-meta-root', type=str, required=True,
                        help='Root dir containing scene metadata JSONs (e.g. .../test/metadata)')
    parser.add_argument('--envmaps-root', type=str, required=True,
                        help='Root dir containing per-scene envmaps (e.g. .../test/envmaps)')
    parser.add_argument('--mesh-root', type=str, required=True,
                        help='Root dir of optimized meshes (e.g. out/polyhaven)')
    parser.add_argument('--output-dir', type=str, default='relight_output')

    parser.add_argument('--res', type=int, default=512,
                        help='Render resolution (overridden by actual image size if available)')
    parser.add_argument('--spp', type=int, default=1)
    parser.add_argument('--n-samples', type=int, default=32,
                        help='Number of light samples for shading')
    parser.add_argument('--probe-res', type=int, default=256,
                        help='Resolution to resize envmap probe (None to keep original)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    glctx = dr.RasterizeGLContext()

    if args.relight_meta is not None:
        meta_files = [args.relight_meta]
    else:
        meta_files = sorted(glob.glob(os.path.join(args.relight_meta_dir, "*.json")))
        print(f"Found {len(meta_files)} relight metadata files")

    success, skip, fail = 0, 0, 0
    for meta_path in meta_files:
        name = os.path.splitext(os.path.basename(meta_path))[0]
        print(f"\n[{name}]")
        try:
            ok = relight_scene(meta_path, args.scene_meta_root, args.envmaps_root,
                               args.mesh_root, args.output_dir, args, glctx)
            if ok:
                success += 1
            else:
                skip += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            fail += 1

    print(f"\nRelighting complete: {success} success, {skip} skipped, {fail} failed")


if __name__ == "__main__":
    main()
