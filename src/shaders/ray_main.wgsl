@group(0) @binding(4) var color_img : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var depth_img : texture_storage_2d<r32float, write>;

@group(1) @binding(0) var depth_tex       : texture_2d<f32>;
@group(1) @binding(1) var godray_hist_tex : texture_2d<f32>;
@group(1) @binding(2) var godray_out      : texture_storage_2d<rgba16float, write>;

@group(2) @binding(0) var color_tex  : texture_2d<f32>;
@group(2) @binding(1) var godray_tex : texture_2d<f32>;
@group(2) @binding(2) var out_img    : texture_storage_2d<rgba16float, write>;

fn tonemap_exp(hdr: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(1.0) - exp(-hdr * POST_EXPOSURE);
}

fn godray_sample_bilerp(px_full: vec2<f32>) -> vec3<f32> {
  let q = px_full * 0.25;
  let q0 = vec2<i32>(i32(floor(q.x)), i32(floor(q.y)));
  let f  = fract(q);

  let qdims = textureDimensions(godray_tex);
  let x0 = clamp(q0.x, 0, i32(qdims.x) - 1);
  let y0 = clamp(q0.y, 0, i32(qdims.y) - 1);
  let x1 = min(x0 + 1, i32(qdims.x) - 1);
  let y1 = min(y0 + 1, i32(qdims.y) - 1);

  let c00 = textureLoad(godray_tex, vec2<i32>(x0, y0), 0).xyz;
  let c10 = textureLoad(godray_tex, vec2<i32>(x1, y0), 0).xyz;
  let c01 = textureLoad(godray_tex, vec2<i32>(x0, y1), 0).xyz;
  let c11 = textureLoad(godray_tex, vec2<i32>(x1, y1), 0).xyz;

  let cx0 = mix(c00, c10, f.x);
  let cx1 = mix(c01, c11, f.x);
  return mix(cx0, cx1, f.y);
}

fn godray_integrate(ro: vec3<f32>, rd: vec3<f32>, t_end: f32, jitter: f32) -> vec3<f32> {
  let base = fog_density();
  if (base <= 0.0 || t_end <= 0.0) { return vec3<f32>(0.0); }

  let tmax = min(t_end, GODRAY_MAX_DIST);
  let dt = tmax / f32(GODRAY_STEPS);

  var sum = vec3<f32>(0.0);

  for (var i: u32 = 0u; i < GODRAY_STEPS; i = i + 1u) {
    let ti = (f32(i) + 0.5 + jitter) * dt;
    if (ti <= 0.0) { continue; }

    let p = ro + rd * ti;

    let Tv = fog_transmittance(ro, rd, ti);
    if (Tv < 0.02) { break; }

    // assume sun visible (no extra shadow trace)
    let Ts = 1.0;

    let dens = base * max(exp(-FOG_HEIGHT_FALLOFF * p.y), 0.15);

    let costh = max(dot(rd, SUN_DIR), 0.0);
    // was: pow(costh, 2.0)
    let costh2 = costh * costh;
    let phase = 0.08 + 0.24 * costh2;

    sum += (SUN_COLOR * SUN_INTENSITY) * (dens * dt) * Tv * Ts * phase;
  }

  return sum * GODRAY_STRENGTH;
}

fn shade_voxel(p: vec3<f32>, n: vec3<f32>, rd: vec3<f32>, mat: u32) -> vec3<f32> {
  let sky = sky_color(rd);

  var albedo = vec3<f32>(0.25, 0.45, 0.22);
  if (mat == 2u) { albedo = vec3<f32>(0.35, 0.35, 0.36); }
  if (mat == 3u) { albedo = vec3<f32>(0.55, 0.50, 0.36); }

  let ambient = 0.18 + 0.42 * clamp(0.5 * (n.y + 1.0), 0.0, 1.0);
  let ndl = max(dot(n, SUN_DIR), 0.0);

  // No shadow ray for perf.
  let direct = SUN_COLOR * SUN_INTENSITY * ndl;

  return albedo * (ambient + 0.75 * direct) + 0.06 * sky;
}

@compute @workgroup_size(8, 8, 1)
fn main_primary(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(color_img);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let res = vec2<f32>(f32(dims.x), f32(dims.y));
  let px  = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  let ro = cam.cam_pos.xyz;
  let rd = ray_dir_from_pixel(px, res);

  var col = sky_color(rd);
  var t_scene = FOG_MAX_DIST;

  let hit = trace_world_svo(ro, rd, FOG_MAX_DIST);
  if (hit.hit) {
    t_scene = hit.t;
    let p = ro + rd * hit.t;
    col = shade_voxel(p, hit.n, rd, hit.mat);
  }

  let T = fog_transmittance(ro, rd, t_scene);
  let fogc = fog_color(rd);
  let fog_amt = (1.0 - T) * FOG_PRIMARY_VIS;
  col = mix(col, fogc, fog_amt);

  let ip = vec2<i32>(i32(gid.x), i32(gid.y));
  textureStore(color_img, ip, vec4<f32>(col, 1.0));
  textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn main_godray(@builtin(global_invocation_id) gid: vec3<u32>) {
  let qdims = textureDimensions(godray_out);
  if (gid.x >= qdims.x || gid.y >= qdims.y) { return; }

  let fdims = textureDimensions(depth_tex);
  let ro = cam.cam_pos.xyz;

  let hip = vec2<i32>(i32(gid.x), i32(gid.y));

  let frame_u = u32(floor(cam.params.x * GODRAY_FRAME_FPS));
  let seed = gid.x + 4096u * gid.y + 131071u * frame_u;
  let j = hash_u01(seed) - 0.5;

  let fx = clamp(i32(gid.x) * 4 + 2, 0, i32(fdims.x) - 1);
  let fy = clamp(i32(gid.y) * 4 + 2, 0, i32(fdims.y) - 1);
  let fp = vec2<i32>(fx, fy);

  let t_scene = textureLoad(depth_tex, fp, 0).x;
  let t_end = min(t_scene, GODRAY_MAX_DIST);

  var cur = vec3<f32>(0.0);
  if (t_end > 0.0 && fog_density() > 0.0) {
    let res_full = vec2<f32>(f32(fdims.x), f32(fdims.y));
    let px2 = vec2<f32>(f32(fp.x) + 0.5, f32(fp.y) + 0.5);
    let rd = ray_dir_from_pixel(px2, res_full);
    cur = godray_integrate(ro, rd, t_end, j);
  }

  let hist = textureLoad(godray_hist_tex, hip, 0).xyz;
  let a = 0.55;
  let blended = mix(cur, hist, a);

  textureStore(godray_out, hip, vec4<f32>(blended, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main_composite(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(out_img);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let ip = vec2<i32>(i32(gid.x), i32(gid.y));
  let base = textureLoad(color_tex, ip, 0).xyz;

  let px = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
  let god = godray_sample_bilerp(px);

  let hdr = base + god * COMPOSITE_GOD_SCALE;
  let ldr = tonemap_exp(hdr);

  textureStore(out_img, ip, vec4<f32>(ldr, 1.0));
}
