"""
Build browser-based cloud pose tuner (HTML + Plotly).

Usage:
1) Run stereo/apps/main.py once (creates stereo_alignment_bundle.npz)
2) Run this script
3) Open generated HTML in browser and tune parameters live
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

import numpy as np

from object_config import get_reconstruction_dir, get_pose_params_file

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = get_reconstruction_dir()
BUNDLE_NAME = "stereo_alignment_bundle.npz"
HTML_NAME = "cloud_pose_tuner.html"
MAX_POINTS_PER_FRAME = 0  # 0 = без лимита (все точки по кадрам)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stereo Cloud Pose Tuner</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #111; color: #eee; }
    .wrap { display: grid; grid-template-columns: 360px 1fr; height: 100vh; }
    .panel { overflow: auto; padding: 12px; border-right: 1px solid #333; background: #171717; }
    .plot { height: 100vh; }
    .row { margin-bottom: 10px; }
    .row label { display: block; font-size: 12px; color: #bbb; margin-bottom: 4px; }
    .row input[type="range"] { width: 100%; }
    .inline { display: flex; gap: 8px; align-items: center; }
    .inline input[type="number"] { width: 95px; background: #222; color: #eee; border: 1px solid #444; padding: 4px; }
    .inline select { background: #222; color: #eee; border: 1px solid #444; padding: 4px; }
    .flags { display: flex; gap: 14px; margin: 10px 0; }
    .btns { display: flex; gap: 8px; margin-top: 12px; }
    button { background: #2b67c0; color: #fff; border: 0; padding: 8px 10px; border-radius: 4px; cursor: pointer; }
    button:hover { background: #3a78d6; }
    .muted { color: #9ca3af; font-size: 12px; margin-top: 8px; line-height: 1.4; }
    .stat { margin-top: 8px; font-size: 12px; color: #8bd5ff; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h3 style="margin: 4px 0 10px 0;">Cloud Pose Tuner</h3>

      <h4 style="margin: 14px 0 8px 0; color: #aaa;">Включённые снимки</h4>
      <div id="frame_toggles" class="flags" style="flex-wrap: wrap;"></div>

      <div class="row">
        <label>CAMERA_START_ANGLE_DEG (deg)</label>
        <input id="camera_start_angle_deg" type="range" min="-180" max="180" step="0.01" />
        <div class="inline"><input id="camera_start_angle_deg_num" type="number" step="0.01" /></div>
      </div>
      <div class="row">
        <label>TABLE_ROTATION_STEP (deg/frame)</label>
        <input id="table_rotation_step" type="range" min="0.1" max="72" step="0.01" />
        <div class="inline"><input id="table_rotation_step_num" type="number" step="0.01" /></div>
      </div>
      <div class="row">
        <label>EXTRA_FRAME_ROT_Z_DEG (deg/frame)</label>
        <input id="extra_frame_rot_z_deg" type="range" min="-20" max="20" step="0.01" />
        <div class="inline"><input id="extra_frame_rot_z_deg_num" type="number" step="0.01" /></div>
      </div>
      <div class="row">
        <label>PLATFORM_ROTATION_SIGN</label>
        <div class="inline">
          <select id="platform_rotation_sign">
            <option value="-1">-1</option>
            <option value="1">+1</option>
          </select>
        </div>
      </div>

      <div class="row">
        <label><input id="use_turntable" type="checkbox" /> Turntable: камера закреплена, вращение вокруг центра</label>
      </div>
      <div class="muted" style="font-size: 11px; margin: 0 0 8px 0;">
        В Turntable ORBIT_RADIUS/CAMERA_HEIGHT задают положение камеры (для cam→world). TABLE_CENTER — точка, вокруг которой вращаются облака.
      </div>
      <h4 style="margin: 14px 0 8px 0; color: #aaa;">Центр вращения стола (cm)</h4>
      <div class="row">
        <label>TABLE_CENTER_X</label>
        <div class="inline"><input id="table_center_x_num" type="number" step="0.01" /></div>
      </div>
      <div class="row">
        <label>TABLE_CENTER_Y</label>
        <div class="inline"><input id="table_center_y_num" type="number" step="0.01" /></div>
      </div>
      <div class="row">
        <label>TABLE_CENTER_Z</label>
        <div class="inline"><input id="table_center_z_num" type="number" step="0.01" /></div>
      </div>

      <div class="row">
        <label>ORBIT_RADIUS (cm) — радиус орбиты камеры в плоскости XY</label>
        <input id="orbit_radius" type="range" min="5" max="80" step="0.01" />
        <div class="inline"><input id="orbit_radius_num" type="number" step="0.01" /></div>
      </div>
      <div class="row">
        <label>CAMERA_HEIGHT (cm) — высота камеры над столом (ось Z)</label>
        <input id="camera_height" type="range" min="-20" max="40" step="0.01" />
        <div class="inline"><input id="camera_height_num" type="number" step="0.01" /></div>
      </div>
      <div class="row">
        <label>CAMERA_TILT (deg) — наклон камеры вверх/вниз (pitch)</label>
        <input id="camera_tilt_deg" type="range" min="-35" max="35" step="0.01" />
        <div class="inline"><input id="camera_tilt_deg_num" type="number" step="0.01" /></div>
      </div>
      <div class="row">
        <label>CAMERA_OFFSET_Y (cm)</label>
        <input id="camera_offset_y" type="range" min="-20" max="20" step="0.01" />
        <div class="inline"><input id="camera_offset_y_num" type="number" step="0.01" /></div>
      </div>
      <div class="row">
        <label>CAMERA_MIN_DISTANCE (cm, 0=off)</label>
        <input id="camera_min_distance" type="range" min="0" max="200" step="0.01" />
        <div class="inline"><input id="camera_min_distance_num" type="number" step="0.01" /></div>
      </div>
      <div class="row">
        <label>CAMERA_MAX_DISTANCE (cm, 0=off)</label>
        <input id="camera_max_distance" type="range" min="0" max="200" step="0.01" />
        <div class="inline"><input id="camera_max_distance_num" type="number" step="0.01" /></div>
      </div>

      <div class="row">
        <label>CROP_RADIUS (cm)</label>
        <input id="crop_radius" type="range" min="1" max="120" step="0.01" />
        <div class="inline"><input id="crop_radius_num" type="number" step="0.01" /></div>
      </div>
      <div class="row">
        <label>Z_MIN / Z_MAX (cm)</label>
        <div class="inline">
          <input id="z_min_num" type="number" step="0.01" />
          <input id="z_max_num" type="number" step="0.01" />
        </div>
      </div>

      <div class="flags">
        <label><input id="invert_x" type="checkbox" /> invert_x</label>
        <label><input id="use_crop" type="checkbox" /> use_crop</label>
      </div>

      <h4 style="margin: 14px 0 8px 0; color: #aaa;">Цветовой фильтр</h4>
      <div class="row">
        <label><input id="use_color_filter" type="checkbox" /> Включить фильтр по цвету</label>
      </div>
      <div class="row">
        <label>Режим:</label>
        <div>
          <label><input id="color_filter_mode_target" type="radio" name="color_filter_mode" value="target" /> Оставить только цвет (R,G,B ± допуск)</label><br/>
          <label><input id="color_filter_mode_dark" type="radio" name="color_filter_mode" value="exclude_dark" /> Исключить только тёмные (порог яркости)</label>
        </div>
      </div>
      <div id="color_filter_target_row" class="row">
        <label>Целевой цвет RGB (0–255)</label>
        <div class="inline">
          <input id="target_r_num" type="number" min="0" max="255" title="R" />
          <input id="target_g_num" type="number" min="0" max="255" title="G" />
          <input id="target_b_num" type="number" min="0" max="255" title="B" />
        </div>
      </div>
      <div id="color_filter_tolerance_row" class="row">
        <label>Допуск по каналу (0–255)</label>
        <div class="inline"><input id="color_tolerance_num" type="number" min="0" max="255" /></div>
      </div>
      <div id="color_filter_dark_row" class="row" style="display:none;">
        <label>Порог «тёмного» (0–255): отсечь точки, где R,G,B всё &lt;= порога</label>
        <div class="inline"><input id="dark_threshold_num" type="number" min="0" max="255" /></div>
      </div>

      <div class="btns">
        <button id="btn_save">Download JSON</button>
        <button id="btn_save_common">Save Common Params</button>
        <button id="btn_reset">Reset view</button>
      </div>
      <div id="stats" class="stat"></div>
      <div class="muted">
        Built from stereo_alignment_bundle.npz (no disparity recompute).<br/>
        <b>Save Common Params</b> сохраняет параметры в браузере и они автоматически применяются в следующих HTML.<br/>
        Download JSON: сохраните файл в <b>Configs/pose_tuned_params.json</b> для использования в Python-скриптах.
      </div>
    </div>
    <div id="plot" class="plot"></div>
  </div>

  <script>
    const DATA = __DATA_JSON__;
    const COMMON_PARAMS_STORAGE_KEY = "stereo_pose_common_params_v1";
    const plotEl = document.getElementById("plot");
    const statsEl = document.getElementById("stats");
    const RENDER_MAX = 500000;  // Cap points to avoid Plotly "Invalid typed array length" on large clouds
    let userCamera = {eye: {x: 1.5, y: 1.5, z: 1.0}};
    let relayoutBound = false;

    const params = {
      camera_start_angle_deg: DATA.defaults.camera_start_angle_deg ?? 0.0,
      table_rotation_step: DATA.defaults.rotation_step_deg,
      extra_frame_rot_z_deg: DATA.defaults.extra_frame_rot_z_deg,
      platform_rotation_sign: DATA.defaults.platform_rotation_sign,
      frame_enabled: DATA.defaults.frame_enabled && typeof DATA.defaults.frame_enabled === "object" ? DATA.defaults.frame_enabled : {},
      use_turntable: DATA.defaults.use_turntable ?? true,
      table_center_x: DATA.defaults.table_center_x ?? 0.0,
      table_center_y: DATA.defaults.table_center_y ?? 0.0,
      table_center_z: DATA.defaults.table_center_z ?? 0.0,
      orbit_radius: DATA.defaults.orbit_radius,
      camera_height: DATA.defaults.camera_height,
      camera_tilt_deg: DATA.defaults.camera_tilt_deg,
      camera_offset_y: DATA.defaults.camera_offset_y,
      camera_min_distance: DATA.defaults.camera_min_distance,
      camera_max_distance: DATA.defaults.camera_max_distance,
      crop_radius: 30.0,
      z_min: -2.0,
      z_max: 15.0,
      invert_x: true,
      use_crop: true,
      use_color_filter: false,
      color_filter_mode: "exclude_dark",
      target_r: 255,
      target_g: 50,
      target_b: 50,
      color_tolerance: 150,
      dark_threshold: 40,
    };

    function buildPosePayload() {
      return {
        FRAME_ENABLED: params.frame_enabled,
        USE_TURNTABLE: params.use_turntable,
        TABLE_CENTER_X: params.table_center_x,
        TABLE_CENTER_Y: params.table_center_y,
        TABLE_CENTER_Z: params.table_center_z,
        CAMERA_START_ANGLE_DEG: params.camera_start_angle_deg,
        TABLE_ROTATION_STEP: params.table_rotation_step,
        EXTRA_FRAME_ROT_Z_DEG: params.extra_frame_rot_z_deg,
        PLATFORM_ROTATION_SIGN: params.platform_rotation_sign,
        ORBIT_RADIUS: params.orbit_radius,
        CAMERA_HEIGHT: params.camera_height,
        CAMERA_TILT: params.camera_tilt_deg,
        CAMERA_OFFSET_Y: params.camera_offset_y,
        CAMERA_MIN_DISTANCE: params.camera_min_distance,
        CAMERA_MAX_DISTANCE: params.camera_max_distance,
        USE_GEOMETRIC_CROP: params.use_crop,
        CROP_RADIUS: params.crop_radius,
        Z_MIN: params.z_min,
        Z_MAX: params.z_max,
        INVERT_X_FINAL: params.invert_x,
        USE_COLOR_FILTER: params.use_color_filter,
        COLOR_FILTER_MODE: params.color_filter_mode || "target",
        TARGET_COLOR_RGB: [params.target_r, params.target_g, params.target_b],
        COLOR_TOLERANCE: params.color_tolerance,
        DARK_THRESHOLD: params.dark_threshold ?? 40
      };
    }

    function applyCommonParamsFromStorage() {
      try {
        const raw = localStorage.getItem(COMMON_PARAMS_STORAGE_KEY);
        if (!raw) return;
        const saved = JSON.parse(raw);
        if (!saved || typeof saved !== "object") return;
        if (saved.USE_TURNTABLE !== undefined) params.use_turntable = !!saved.USE_TURNTABLE;
        if (saved.TABLE_CENTER_X !== undefined) params.table_center_x = Number(saved.TABLE_CENTER_X);
        if (saved.TABLE_CENTER_Y !== undefined) params.table_center_y = Number(saved.TABLE_CENTER_Y);
        if (saved.TABLE_CENTER_Z !== undefined) params.table_center_z = Number(saved.TABLE_CENTER_Z);
        if (saved.CAMERA_START_ANGLE_DEG !== undefined) params.camera_start_angle_deg = Number(saved.CAMERA_START_ANGLE_DEG);
        if (saved.TABLE_ROTATION_STEP !== undefined) params.table_rotation_step = Number(saved.TABLE_ROTATION_STEP);
        if (saved.EXTRA_FRAME_ROT_Z_DEG !== undefined) params.extra_frame_rot_z_deg = Number(saved.EXTRA_FRAME_ROT_Z_DEG);
        if (saved.PLATFORM_ROTATION_SIGN !== undefined) params.platform_rotation_sign = Number(saved.PLATFORM_ROTATION_SIGN);
        if (saved.ORBIT_RADIUS !== undefined) params.orbit_radius = Number(saved.ORBIT_RADIUS);
        if (saved.CAMERA_HEIGHT !== undefined) params.camera_height = Number(saved.CAMERA_HEIGHT);
        if (saved.CAMERA_TILT !== undefined) params.camera_tilt_deg = Number(saved.CAMERA_TILT);
        if (saved.CAMERA_OFFSET_Y !== undefined) params.camera_offset_y = Number(saved.CAMERA_OFFSET_Y);
        if (saved.CAMERA_MIN_DISTANCE !== undefined) params.camera_min_distance = Number(saved.CAMERA_MIN_DISTANCE);
        if (saved.CAMERA_MAX_DISTANCE !== undefined) params.camera_max_distance = Number(saved.CAMERA_MAX_DISTANCE);
        if (saved.USE_GEOMETRIC_CROP !== undefined) params.use_crop = !!saved.USE_GEOMETRIC_CROP;
        if (saved.CROP_RADIUS !== undefined) params.crop_radius = Number(saved.CROP_RADIUS);
        if (saved.Z_MIN !== undefined) params.z_min = Number(saved.Z_MIN);
        if (saved.Z_MAX !== undefined) params.z_max = Number(saved.Z_MAX);
        if (saved.INVERT_X_FINAL !== undefined) params.invert_x = !!saved.INVERT_X_FINAL;
        if (saved.USE_COLOR_FILTER !== undefined) params.use_color_filter = !!saved.USE_COLOR_FILTER;
        if (saved.COLOR_FILTER_MODE !== undefined) params.color_filter_mode = String(saved.COLOR_FILTER_MODE);
        if (Array.isArray(saved.TARGET_COLOR_RGB) && saved.TARGET_COLOR_RGB.length >= 3) {
          params.target_r = Number(saved.TARGET_COLOR_RGB[0]);
          params.target_g = Number(saved.TARGET_COLOR_RGB[1]);
          params.target_b = Number(saved.TARGET_COLOR_RGB[2]);
        }
        if (saved.COLOR_TOLERANCE !== undefined) params.color_tolerance = Number(saved.COLOR_TOLERANCE);
        if (saved.DARK_THRESHOLD !== undefined) params.dark_threshold = Number(saved.DARK_THRESHOLD);
      } catch (_) {}
    }

    function bindRangeWithNumber(id, onChange) {
      const r = document.getElementById(id);
      const n = document.getElementById(id + "_num");
      r.value = String(params[id]);
      n.value = String(params[id]);
      r.addEventListener("input", () => {
        params[id] = Number(r.value);
        n.value = r.value;
        onChange();
      });
      n.addEventListener("change", () => {
        params[id] = Number(n.value);
        r.value = String(params[id]);
        onChange();
      });
    }

    function bindNumberOnly(id, key, onChange) {
      const el = document.getElementById(id);
      el.value = String(params[key]);
      el.addEventListener("change", () => {
        params[key] = Number(el.value);
        onChange();
      });
    }

    function initControls(onChange) {
      bindRangeWithNumber("camera_start_angle_deg", onChange);
      bindRangeWithNumber("table_rotation_step", onChange);
      bindRangeWithNumber("extra_frame_rot_z_deg", onChange);
      bindRangeWithNumber("orbit_radius", onChange);
      bindRangeWithNumber("camera_height", onChange);
      bindRangeWithNumber("camera_tilt_deg", onChange);
      bindRangeWithNumber("camera_offset_y", onChange);
      bindRangeWithNumber("camera_min_distance", onChange);
      bindRangeWithNumber("camera_max_distance", onChange);
      bindRangeWithNumber("crop_radius", onChange);

      bindNumberOnly("z_min_num", "z_min", onChange);
      bindNumberOnly("z_max_num", "z_max", onChange);

      const signSel = document.getElementById("platform_rotation_sign");
      signSel.value = String(params.platform_rotation_sign);
      signSel.addEventListener("change", () => {
        params.platform_rotation_sign = Number(signSel.value);
        onChange();
      });

      const invert = document.getElementById("invert_x");
      const useCrop = document.getElementById("use_crop");
      const useTurntable = document.getElementById("use_turntable");
      invert.checked = params.invert_x;
      useCrop.checked = params.use_crop;
      useTurntable.checked = params.use_turntable;
      invert.addEventListener("change", () => { params.invert_x = invert.checked; onChange(); });
      useCrop.addEventListener("change", () => { params.use_crop = useCrop.checked; onChange(); });
      useTurntable.addEventListener("change", () => { params.use_turntable = useTurntable.checked; onChange(); });
      const frameTogglesEl = document.getElementById("frame_toggles");
      for (const f of DATA.frames) {
        if (params.frame_enabled[f.idx] === undefined) params.frame_enabled[f.idx] = true;
        const cb = document.createElement("label");
        cb.innerHTML = `<input id="frame_${f.idx}" type="checkbox" ${params.frame_enabled[f.idx] ? "checked" : ""} /> frame ${f.idx}`;
        cb.style.marginRight = "12px";
        cb.querySelector("input").addEventListener("change", () => {
          params.frame_enabled[f.idx] = cb.querySelector("input").checked;
          onChange();
        });
        frameTogglesEl.appendChild(cb);
      }
      bindNumberOnly("table_center_x_num", "table_center_x", onChange);
      bindNumberOnly("table_center_y_num", "table_center_y", onChange);
      bindNumberOnly("table_center_z_num", "table_center_z", onChange);
      document.getElementById("table_center_x_num").value = String(params.table_center_x);
      document.getElementById("table_center_y_num").value = String(params.table_center_y);
      document.getElementById("table_center_z_num").value = String(params.table_center_z);

      const useColorFilter = document.getElementById("use_color_filter");
      useColorFilter.checked = params.use_color_filter;
      useColorFilter.addEventListener("change", () => { params.use_color_filter = useColorFilter.checked; onChange(); });
      function updateColorFilterRows() {
        const mode = params.color_filter_mode || "target";
        document.getElementById("color_filter_target_row").style.display = mode === "target" ? "" : "none";
        document.getElementById("color_filter_tolerance_row").style.display = mode === "target" ? "" : "none";
        document.getElementById("color_filter_dark_row").style.display = mode === "exclude_dark" ? "" : "none";
      }
      document.getElementById("color_filter_mode_target").checked = (params.color_filter_mode || "target") === "target";
      document.getElementById("color_filter_mode_dark").checked = (params.color_filter_mode || "target") === "exclude_dark";
      document.getElementById("color_filter_mode_target").addEventListener("change", () => { params.color_filter_mode = "target"; updateColorFilterRows(); onChange(); });
      document.getElementById("color_filter_mode_dark").addEventListener("change", () => { params.color_filter_mode = "exclude_dark"; updateColorFilterRows(); onChange(); });
      bindNumberOnly("target_r_num", "target_r", onChange);
      bindNumberOnly("target_g_num", "target_g", onChange);
      bindNumberOnly("target_b_num", "target_b", onChange);
      bindNumberOnly("color_tolerance_num", "color_tolerance", onChange);
      bindNumberOnly("dark_threshold_num", "dark_threshold", onChange);
      document.getElementById("target_r_num").value = String(params.target_r);
      document.getElementById("target_g_num").value = String(params.target_g);
      document.getElementById("target_b_num").value = String(params.target_b);
      document.getElementById("color_tolerance_num").value = String(params.color_tolerance);
      document.getElementById("dark_threshold_num").value = String(params.dark_threshold ?? 40);
      updateColorFilterRows();

      document.getElementById("btn_save").addEventListener("click", saveJson);
      document.getElementById("btn_save_common").addEventListener("click", saveCommonParams);
      document.getElementById("btn_reset").addEventListener("click", () => {
        userCamera = {eye: {x: 1.5, y: 1.5, z: 1.0}};
        Plotly.relayout(plotEl, {"scene.camera": userCamera});
      });
    }

    function normalize3(v) {
      const n = Math.hypot(v[0], v[1], v[2]);
      if (n < 1e-12) return null;
      return [v[0] / n, v[1] / n, v[2] / n];
    }

    function cross3(a, b) {
      return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
      ];
    }

    function matMul3(a, b) {
      const out = [[0,0,0],[0,0,0],[0,0,0]];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
      }
      return out;
    }

    // Угол кадра в радианах: start + sign*step*idx + extra*idx (как в main.py)
    function frameAngleRad(frameIdx) {
      const startRad = params.camera_start_angle_deg * Math.PI / 180.0;
      const stepRad = params.table_rotation_step * Math.PI / 180.0;
      const extraRad = params.extra_frame_rot_z_deg * Math.PI / 180.0;
      return startRad + params.platform_rotation_sign * stepRad * frameIdx + extraRad * frameIdx;
    }

    // Поза камеры для заданного угла орбиты (радианы). Камера на окружности смотрит в центр.
    function buildCameraPose(angleRad) {
      if (angleRad === undefined) angleRad = 0.0;
      const R = params.orbit_radius;
      const cBlock = [
        R * Math.cos(angleRad),
        R * Math.sin(angleRad) + params.camera_offset_y,
        params.camera_height
      ];
      const toTarget = normalize3([-cBlock[0], -cBlock[1], -cBlock[2]]) || [-1, 0, 0];
      const up = [0, 0, 1];
      let xAxis = normalize3(cross3(toTarget, up));
      if (!xAxis) xAxis = [0, -1, 0];
      let yUp = normalize3(cross3(xAxis, toTarget));
      if (!yUp) yUp = [0, 1, 0];
      const yDown = [-yUp[0], -yUp[1], -yUp[2]];
      let rBase = [
        [xAxis[0], yDown[0], toTarget[0]],
        [xAxis[1], yDown[1], toTarget[1]],
        [xAxis[2], yDown[2], toTarget[2]]
      ];
      const t = params.camera_tilt_deg * Math.PI / 180.0;
      if (Math.abs(t) > 1e-12) {
        const c = Math.cos(t), s = Math.sin(t);
        const rx = [[1, 0, 0], [0, c, -s], [0, s, c]];
        rBase = matMul3(rBase, rx);
      }
      return {R: rBase, C: cBlock};
    }

    function camToWorld(pt, pose) {
      const x = pt[0], y = pt[1], z = pt[2];
      return [
        pose.C[0] + pose.R[0][0] * x + pose.R[0][1] * y + pose.R[0][2] * z,
        pose.C[1] + pose.R[1][0] * x + pose.R[1][1] * y + pose.R[1][2] * z,
        pose.C[2] + pose.R[2][0] * x + pose.R[2][1] * y + pose.R[2][2] * z
      ];
    }

    function rotateAroundZ(p, angle, center) {
      const c = Math.cos(angle), s = Math.sin(angle);
      const dx = p[0] - center[0];
      const dy = p[1] - center[1];
      return [
        c * dx - s * dy + center[0],
        s * dx + c * dy + center[1],
        p[2]
      ];
    }

    function buildCloud() {
      const pivot = [params.table_center_x, params.table_center_y, params.table_center_z];
      const xs = [], ys = [], zs = [];
      const colors = [];
      const hasColors = DATA.has_colors || false;
      let totalRaw = 0;
      // Turntable: одна поза камеры (кадр 0), вращение вокруг pivot
      const pose0 = buildCameraPose(params.camera_start_angle_deg * Math.PI / 180.0);
      for (const frame of DATA.frames) {
        if (params.frame_enabled[frame.idx] === false) continue;
        const pose = params.use_turntable ? pose0 : buildCameraPose(frameAngleRad(frame.idx));
        const angleUndoRad = params.use_turntable
          ? -(params.platform_rotation_sign * params.table_rotation_step * Math.PI / 180.0 * frame.idx)
          : 0;
        const pts = frame.pts;
        const frameColors = frame.colors || null;
        totalRaw += pts.length;
        for (let i = 0; i < pts.length; i++) {
          const dCam = Math.hypot(pts[i][0], pts[i][1], pts[i][2]);
          if (params.camera_min_distance > 0 && dCam < params.camera_min_distance) {
            continue;
          }
          if (params.camera_max_distance > 0 && dCam > params.camera_max_distance) {
            continue;
          }
          let p = camToWorld(pts[i], pose);
          if (params.use_turntable && Math.abs(angleUndoRad) > 1e-12) {
            p = rotateAroundZ(p, angleUndoRad, pivot);
          }
          if (params.invert_x) p[0] = -p[0];
          if (params.use_crop) {
            const rx = p[0] - pivot[0];
            const ry = p[1] - pivot[1];
            const rxy = Math.hypot(rx, ry);
            if (rxy >= params.crop_radius || p[2] <= params.z_min || p[2] >= params.z_max) {
              continue;
            }
          }
          if (params.use_color_filter && hasColors && frameColors && i < frameColors.length) {
            const rgb = frameColors[i];
            const mode = params.color_filter_mode || "target";
            if (mode === "exclude_dark") {
              const th = params.dark_threshold ?? 40;
              if (rgb[0] <= th && rgb[1] <= th && rgb[2] <= th) continue;
            } else {
              const tr = params.target_r, tg = params.target_g, tb = params.target_b;
              const tol = params.color_tolerance;
              if (Math.abs(rgb[0] - tr) > tol || Math.abs(rgb[1] - tg) > tol || Math.abs(rgb[2] - tb) > tol) {
                continue;
              }
            }
          }
          xs.push(p[0]); ys.push(p[1]); zs.push(p[2]);
          if (hasColors && frameColors && i < frameColors.length) {
            const rgb = frameColors[i];
            colors.push(`rgb(${rgb[0]},${rgb[1]},${rgb[2]})`);
          }
        }
      }
      const n = xs.length;
      if (RENDER_MAX <= 0 || n <= RENDER_MAX) {
        return {x: xs, y: ys, z: zs, colors: colors.length > 0 ? colors : null, nRaw: totalRaw, nOut: n, nRender: n};
      }
      const step = Math.ceil(n / RENDER_MAX);
      const xr = [], yr = [], zr = [], colorsR = [];
      for (let i = 0; i < n; i += step) {
        xr.push(xs[i]); yr.push(ys[i]); zr.push(zs[i]);
        if (colors.length > 0 && i < colors.length) colorsR.push(colors[i]);
      }
      return {x: xr, y: yr, z: zr, colors: colorsR.length > 0 ? colorsR : null, nRaw: totalRaw, nOut: n, nRender: xr.length};
    }

    function render() {
      const cloud = buildCloud();
      statsEl.textContent = `raw=${cloud.nRaw} | after filter=${cloud.nOut} | render=${cloud.nRender}`;
      let xMin = 0, xMax = 1, yMin = 0, yMax = 1, zMin = 0, zMax = 1;
      if (cloud.nRender > 0) {
        xMin = xMax = cloud.x[0];
        yMin = yMax = cloud.y[0];
        zMin = zMax = cloud.z[0];
        for (let i = 1; i < cloud.nRender; i++) {
          const x = cloud.x[i], y = cloud.y[i], z = cloud.z[i];
          if (x < xMin) xMin = x;
          if (x > xMax) xMax = x;
          if (y < yMin) yMin = y;
          if (y > yMax) yMax = y;
          if (z < zMin) zMin = z;
          if (z > zMax) zMax = z;
        }
      }
      const maxRange = Math.max(xMax - xMin, yMax - yMin, zMax - zMin, 1.0);
      const half = maxRange / 2.0;
      const cx = (xMin + xMax) / 2.0;
      const cy = (yMin + yMax) / 2.0;
      const cz = (zMin + zMax) / 2.0;
      const markerConfig = {size: 2};
      if (cloud.colors && cloud.colors.length > 0) {
        markerConfig.color = cloud.colors;
      } else {
        markerConfig.color = "#4ea1ff";
      }
      const data = [{
        type: "scatter3d",
        mode: "markers",
        x: cloud.x, y: cloud.y, z: cloud.z,
        marker: markerConfig,
        name: "Points"
      }];
      const layout = {
        paper_bgcolor: "#111",
        plot_bgcolor: "#111",
        margin: {l: 0, r: 0, b: 0, t: 30},
        title: "Stereo Cloud Pose Tuner",
        uirevision: "keep-camera",
        scene: {
          xaxis: {title: "X (cm)", color: "#ccc", range: [cx - half, cx + half]},
          yaxis: {title: "Y (cm)", color: "#ccc", range: [cy - half, cy + half]},
          zaxis: {title: "Z (cm)", color: "#ccc", range: [cz - half, cz + half]},
          aspectmode: "cube",
          camera: userCamera
        }
      };
      Plotly.react(plotEl, data, layout, {displaylogo: false, responsive: true}).then(() => {
        if (!relayoutBound) {
          plotEl.on("plotly_relayout", (ev) => {
            if (ev && ev["scene.camera"]) {
              userCamera = ev["scene.camera"];
            }
          });
          relayoutBound = true;
        }
      });
    }

    let timer = null;
    function scheduleRender() {
      if (timer) clearTimeout(timer);
      timer = setTimeout(render, 30);
    }

    function saveJson() {
      const out = buildPosePayload();
      const blob = new Blob([JSON.stringify(out, null, 2)], {type: "application/json"});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "pose_tuned_params.json";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    function saveCommonParams() {
      try {
        localStorage.setItem(COMMON_PARAMS_STORAGE_KEY, JSON.stringify(buildPosePayload()));
        statsEl.textContent = "Common params saved. They will be auto-applied in next tuner HTML files.";
      } catch (_) {
        statsEl.textContent = "Failed to save common params in browser storage.";
      }
    }

    applyCommonParamsFromStorage();
    initControls(scheduleRender);
    render();
  </script>
</body>
</html>
"""


def downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[idx]


def resolve_bundle_and_output_dir() -> tuple[Path, Path]:
    """Find bundle produced by main.py across current and legacy output layouts."""
    candidates = [
        OUTPUT_DIR / BUNDLE_NAME,  # object_config layout
        ROOT_DIR / "outputs" / "reconstruction" / BUNDLE_NAME,  # main.py layout
        ROOT_DIR / "outputs" / "tri" / "reconstruction" / BUNDLE_NAME,  # legacy object-specific layout
    ]
    for bundle_path in candidates:
        if bundle_path.exists():
            return bundle_path, bundle_path.parent
    # Default to object_config location when bundle is missing (for clear error path).
    return candidates[0], OUTPUT_DIR


def load_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Bundle not found: {path}. Run apps/main.py first.")
    data = np.load(path, allow_pickle=True)
    frame_indices = data["frame_indices"].astype(np.int32).tolist()
    clouds = [np.asarray(c, dtype=np.float32) for c in data["clouds_cam"].tolist()]
    
    # Загружаем цвета, если они есть
    has_colors = "colors_rgb" in data.files
    colors_list = None
    if has_colors:
        colors_list = [np.asarray(c, dtype=np.uint8) if c is not None else None for c in data["colors_rgb"].tolist()]

    frames = []
    for i, (idx, pts) in enumerate(zip(frame_indices, clouds)):
        pts_ds = downsample(pts, MAX_POINTS_PER_FRAME).astype(np.float32)
        frame_data = {"idx": int(idx), "pts": pts_ds.tolist()}
        # Добавляем цвета, если они есть и соответствуют даунсэмплированным точкам
        if has_colors and colors_list[i] is not None:
            if MAX_POINTS_PER_FRAME > 0 and len(pts) > MAX_POINTS_PER_FRAME:
                idx_ds = np.linspace(0, len(pts) - 1, MAX_POINTS_PER_FRAME, dtype=int)
                colors_ds = colors_list[i][idx_ds]
            else:
                colors_ds = colors_list[i]
            frame_data["colors"] = colors_ds.tolist()
        frames.append(frame_data)

    defaults = {
        "rotation_step_deg": float(data["rotation_step_deg"]),
        "camera_start_angle_deg": float(data["camera_start_angle_deg"]) if "camera_start_angle_deg" in data.files else 0.0,
        "extra_frame_rot_z_deg": float(data["extra_frame_rot_z_deg"]) if "extra_frame_rot_z_deg" in data.files else 0.0,
        "platform_rotation_sign": int(data["platform_rotation_sign"]),
        "orbit_radius": float(data["orbit_radius"]),
        "camera_height": float(data["camera_height"]),
        "camera_tilt_deg": float(data["camera_tilt_deg"]),
        "camera_offset_y": float(data["camera_offset_y"]),
        "camera_min_distance": float(data["camera_min_distance"]) if "camera_min_distance" in data.files else 0.0,
        "camera_max_distance": float(data["camera_max_distance"]) if "camera_max_distance" in data.files else 0.0,
        "use_turntable": True,
        "table_center_x": 0.0,
        "table_center_y": 0.0,
        "table_center_z": 0.0,
        "frame_enabled": {},
    }
    return {"frames": frames, "has_colors": has_colors, "defaults": defaults}


def main() -> None:
    bundle_file, output_dir = resolve_bundle_and_output_dir()
    bundle = load_bundle(bundle_file)
    pose_file = get_pose_params_file()
    if pose_file.exists():
        try:
            with pose_file.open("r", encoding="utf-8") as f:
                pose = json.load(f)
            for key, js_key in [
                ("frame_enabled", "FRAME_ENABLED"),
                ("use_turntable", "USE_TURNTABLE"),
                ("table_center_x", "TABLE_CENTER_X"),
                ("table_center_y", "TABLE_CENTER_Y"),
                ("table_center_z", "TABLE_CENTER_Z"),
                ("camera_start_angle_deg", "CAMERA_START_ANGLE_DEG"),
                ("rotation_step_deg", "TABLE_ROTATION_STEP"),
                ("extra_frame_rot_z_deg", "EXTRA_FRAME_ROT_Z_DEG"),
                ("platform_rotation_sign", "PLATFORM_ROTATION_SIGN"),
                ("orbit_radius", "ORBIT_RADIUS"),
                ("camera_height", "CAMERA_HEIGHT"),
                ("camera_tilt_deg", "CAMERA_TILT"),
                ("camera_offset_y", "CAMERA_OFFSET_Y"),
                ("camera_min_distance", "CAMERA_MIN_DISTANCE"),
                ("camera_max_distance", "CAMERA_MAX_DISTANCE"),
            ]:
                if js_key in pose:
                    bundle["defaults"][key] = pose[js_key]
        except Exception:
            pass
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(bundle, ensure_ascii=False))
    html_file = output_dir / HTML_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    html_file.write_text(html, encoding="utf-8")
    print(f"Built: {html_file}")

if __name__ == "__main__":
    main()

















