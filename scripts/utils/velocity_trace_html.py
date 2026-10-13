# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

"""Reusable HTML renderer for Wheelbipe velocity/reward trace CSV data."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence


def build_reward_signs(reward_scales: Mapping[str, float] | None = None, rows: Sequence[Mapping] | None = None) -> dict[str, int]:
    """Build reward column signs from cfg reward scales, with data fallback for offline CSV export."""
    signs: dict[str, int] = {"reward_total": 0}
    if reward_scales:
        for key, weight in reward_scales.items():
            value = float(weight)
            signs[f"reward_{key}"] = 1 if value > 0.0 else -1 if value < 0.0 else 0
    if rows:
        for key in rows[0].keys():
            if key.startswith("reward_") and key not in signs:
                signs[key] = 0
    return signs


def build_velocity_trace_html(rows: Sequence[Mapping], reward_signs: Mapping[str, int] | None = None) -> str:
    """Render the self-contained velocity/reward analysis HTML."""
    rows_json = json.dumps(list(rows), ensure_ascii=False)
    reward_signs_json = json.dumps(dict(reward_signs or build_reward_signs(rows=rows)), ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Wheelbipe Velocity Trace</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; background: #050607; color: #e8edf2; overflow-y: auto; }}
.bar {{ padding: 10px 14px; background: #0d1117; color: #e8edf2; display: flex; gap: 18px; align-items: center; flex-wrap: wrap; border-bottom: 1px solid #263241; }}
.bar label {{ display: inline-flex; gap: 6px; align-items: center; }}
button {{ background: #1f2937; color: #e8edf2; border: 1px solid #3b4656; border-radius: 4px; padding: 4px 9px; }}
#plot {{ display: block; width: 100vw; min-height: calc(100vh - 52px); background: #050607; cursor: grab; }}
#plot:active {{ cursor: grabbing; }}
</style>
</head>
<body>
<div class="bar">
  <strong>Velocity Trace</strong>
  <span id="meta"></span>
  <label><input type="checkbox" data-series="cmd_x" checked>cmd_x</label>
  <label><input type="checkbox" data-series="vel_x_b" checked>vel_x_b</label>
  <label><input type="checkbox" data-series="cmd_yaw" checked>cmd_yaw</label>
  <label><input type="checkbox" data-series="yaw_rate_b" checked>yaw_rate_b</label>
  <label><input type="checkbox" data-series="height_cmd" checked>height_cmd</label>
  <label><input type="checkbox" data-series="height_reward_ref" checked>height</label>
  <button id="reset">Reset zoom</button>
</div>
<canvas id="plot"></canvas>
<script>
const rows = {rows_json};
const colors = {{
  cmd_x:"#d73027",
  vel_x_b:"#4575b4",
  cmd_yaw:"#fdae61",
  yaw_rate_b:"#1a9850",
  height_cmd:"#c77dff",
  height_obs:"#ff7ab6",
  height_relative:"#8bd3ff",
  height_reward_ref:"#f2cc60",
}};
const rewardKeys = rows.length
  ? ["reward_total", ...Object.keys(rows[0]).filter(k => k.startsWith("reward_") && k !== "reward_total")]
  : [];
const rewardSigns = {reward_signs_json};
let visible = {{cmd_x:true, vel_x_b:true, cmd_yaw:true, yaw_rate_b:true, height_cmd:true, height_reward_ref:true}};
let xMin = rows.length ? rows[0].sim_time_s : 0;
let xMax = rows.length ? rows[rows.length - 1].sim_time_s : 1;
const xFull = [xMin, xMax];
const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");
let dragging = false, lastX = 0;
let hoverTime = null, hoverRow = null, hoverCanvasX = null;

document.getElementById("meta").textContent = rows.length
  ? `rows=${{rows.length}} env=${{rows[rows.length-1].env_id}} terrain=${{rows[rows.length-1].terrain}}`
  : "no data";

function resize() {{
  const rowCssH = 22;
  const minLineCssH = 480;
  const heatCssH = rewardKeys.length ? rewardKeys.length * rowCssH + 76 : 0;
  const targetCssH = Math.max(window.innerHeight - canvas.offsetTop, minLineCssH + heatCssH + 88);
  canvas.style.height = `${{targetCssH}}px`;
  canvas.width = Math.floor(canvas.clientWidth * devicePixelRatio);
  canvas.height = Math.floor(canvas.clientHeight * devicePixelRatio);
  draw();
}}
function selectedRows() {{ return rows.filter(r => r.sim_time_s >= xMin && r.sim_time_s <= xMax); }}
function nearestRow(time) {{
  if (!rows.length) return null;
  let lo = 0, hi = rows.length - 1;
  while (lo < hi) {{
    const mid = Math.floor((lo + hi) / 2);
    if (rows[mid].sim_time_s < time) lo = mid + 1;
    else hi = mid;
  }}
  const a = rows[Math.max(0, lo - 1)], b = rows[lo];
  if (!a) return b;
  if (!b) return a;
  return Math.abs(a.sim_time_s - time) <= Math.abs(b.sim_time_s - time) ? a : b;
}}
function updateHoverFromEvent(e) {{
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * devicePixelRatio;
  const padL = 120 * devicePixelRatio, padR = 24 * devicePixelRatio;
  const width = canvas.width - padL - padR;
  if (x < padL || x > padL + width) {{
    hoverTime = null; hoverRow = null; hoverCanvasX = null;
    return;
  }}
  const ratio = (x - padL) / Math.max(width, 1);
  hoverTime = xMin + ratio * (xMax - xMin);
  hoverRow = nearestRow(hoverTime);
  hoverCanvasX = x;
}}
function yRange(rs, keys) {{
  let vals = [];
  for (const r of rs) for (const k of keys) if (visible[k]) vals.push(r[k]);
  if (!vals.length) return [-1, 1];
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (Math.abs(hi - lo) < 1e-6) {{ lo -= 1; hi += 1; }}
  const pad = 0.08 * (hi - lo);
  return [lo - pad, hi + pad];
}}
function draw() {{
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const padL = 120 * devicePixelRatio, padR = 24 * devicePixelRatio, padT = 20 * devicePixelRatio, padB = 42 * devicePixelRatio;
  const gap = 28 * devicePixelRatio;
  const minRewardRowH = 22 * devicePixelRatio;
  const heatH = rewardKeys.length ? rewardKeys.length * minRewardRowH + 70 * devicePixelRatio : 0;
  const lineH = h - padT - padB - heatH - (rewardKeys.length ? gap : 0);
  const chartGap = 26 * devicePixelRatio;
  const pw = w - padL - padR;
  const chartsH = Math.max(lineH, 440 * devicePixelRatio);
  const velocityH = Math.max(210 * devicePixelRatio, (chartsH - chartGap) * 0.52);
  const heightH = Math.max(170 * devicePixelRatio, chartsH - chartGap - velocityH);
  ctx.fillStyle = "#050607"; ctx.fillRect(0,0,w,h);
  const rs = selectedRows();
  drawLineChart("velocity / yaw", ["cmd_x", "vel_x_b", "cmd_yaw", "yaw_rate_b"], rs, padL, padT, pw, velocityH);
  drawLineChart("height", ["height_cmd", "height_reward_ref", "height_obs", "height_relative"], rs, padL, padT + velocityH + chartGap, pw, heightH);
  drawRewardHeatmap(rs, padL, padT + chartsH + gap, pw, heatH);
  ctx.fillStyle = "#a9b4c2";
  ctx.fillText(`time ${{xMin.toFixed(2)}} - ${{xMax.toFixed(2)}} s`, padL, h - 14*devicePixelRatio);
}}
function drawLineChart(title, keys, rs, x0, y0, width, height) {{
  const [yMin, yMax] = yRange(rs, keys);
  const sx = x => x0 + (x - xMin) / Math.max(xMax - xMin, 1e-9) * width;
  const sy = y => y0 + (yMax - y) / Math.max(yMax - yMin, 1e-9) * height;
  ctx.strokeStyle = "#344154"; ctx.lineWidth = 1 * devicePixelRatio;
  ctx.beginPath(); ctx.rect(x0, y0, width, height); ctx.stroke();
  ctx.fillStyle = "#d7dee8"; ctx.font = `${{13 * devicePixelRatio}}px Arial`;
  ctx.fillText(title, x0 + 8 * devicePixelRatio, y0 + 17 * devicePixelRatio);
  for (let i=0;i<=5;i++) {{
    const y = yMin + (yMax-yMin)*i/5;
    const yy = sy(y);
    ctx.strokeStyle = "#182231"; ctx.beginPath(); ctx.moveTo(x0, yy); ctx.lineTo(x0+width, yy); ctx.stroke();
    ctx.fillStyle = "#a9b4c2"; ctx.font = `${{12 * devicePixelRatio}}px Arial`;
    ctx.fillText(y.toFixed(3), 8*devicePixelRatio, yy+4*devicePixelRatio);
  }}
  for (const k of keys) {{
    if (!visible[k] || rs.length < 2) continue;
    ctx.strokeStyle = colors[k]; ctx.lineWidth = 2 * devicePixelRatio; ctx.beginPath();
    rs.forEach((r, i) => {{ const x=sx(r.sim_time_s), y=sy(r[k]); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }});
    ctx.stroke();
  }}
  if (hoverRow && hoverRow.sim_time_s >= xMin && hoverRow.sim_time_s <= xMax) {{
    const hx = sx(hoverRow.sim_time_s);
    ctx.strokeStyle = "#e5edf5aa"; ctx.lineWidth = 1 * devicePixelRatio;
    ctx.beginPath(); ctx.moveTo(hx, y0); ctx.lineTo(hx, y0 + height); ctx.stroke();
    for (const k of keys) {{
      if (!visible[k]) continue;
      const v = Number(hoverRow[k]);
      if (!Number.isFinite(v)) continue;
      ctx.fillStyle = colors[k];
      ctx.beginPath(); ctx.arc(hx, sy(v), 3.5 * devicePixelRatio, 0, Math.PI * 2); ctx.fill();
    }}
    drawLineHoverPanel(keys, hoverRow, x0, y0, width);
  }}
  let lx = x0 + 8*devicePixelRatio, ly = y0 + 37*devicePixelRatio;
  for (const k of keys) if (visible[k]) {{
    ctx.fillStyle = colors[k]; ctx.fillRect(lx, ly-10*devicePixelRatio, 16*devicePixelRatio, 3*devicePixelRatio);
    ctx.fillStyle = "#d7dee8"; ctx.font = `${{12 * devicePixelRatio}}px Arial`;
    ctx.fillText(k, lx + 22*devicePixelRatio, ly);
    lx += Math.max(104, k.length * 8 + 34) * devicePixelRatio;
  }}
}}
function drawLineHoverPanel(keys, row, x0, y0, width) {{
  const activeKeys = keys.filter(k => visible[k]);
  if (!activeKeys.length) return;
  const panelW = 230 * devicePixelRatio;
  const lineH = 16 * devicePixelRatio;
  const panelH = (activeKeys.length + 1) * lineH + 12 * devicePixelRatio;
  const px = x0 + width - panelW - 8 * devicePixelRatio;
  const py = y0 + 8 * devicePixelRatio;
  ctx.fillStyle = "rgba(5, 8, 12, 0.82)";
  ctx.strokeStyle = "#3b4656";
  ctx.lineWidth = 1 * devicePixelRatio;
  ctx.beginPath(); ctx.rect(px, py, panelW, panelH); ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#e8edf2"; ctx.font = `${{12 * devicePixelRatio}}px Arial`;
  ctx.fillText(`t=${{row.sim_time_s.toFixed(3)}}s`, px + 8 * devicePixelRatio, py + 15 * devicePixelRatio);
  activeKeys.forEach((k, i) => {{
    const y = py + (i + 2) * lineH;
    ctx.fillStyle = colors[k] || "#d7dee8";
    ctx.fillText(k, px + 8 * devicePixelRatio, y);
    ctx.fillStyle = "#d7dee8";
    const v = Number(row[k]);
    ctx.fillText(Number.isFinite(v) ? v.toFixed(4) : "n/a", px + 138 * devicePixelRatio, y);
  }});
}}
function heatColor(k, v, rowMinMagnitude, rowMaxMagnitude, minNeg, maxPos) {{
  if (!Number.isFinite(v) || Math.abs(v) < 1e-12) return "#111827";
  const absV = Math.abs(v);
  const rowSpan = rowMaxMagnitude - rowMinMagnitude;
  const tLocal = rowSpan > 1e-12
    ? Math.min(1, Math.max(0, (absV - rowMinMagnitude) / rowSpan))
    : (rowMaxMagnitude > 0 ? Math.min(1, absV / rowMaxMagnitude) : 0);
  const tRow = Math.sqrt(tLocal);
  if (rewardSigns[k] > 0) return `rgb(${{Math.round(23 - 17 * tRow)}},${{Math.round(74 + 62 * tRow)}},${{Math.round(49 - 28 * tRow)}})`;
  if (rewardSigns[k] < 0) return `rgb(${{Math.round(88 + 82 * tRow)}},${{Math.round(34 - 26 * tRow)}},${{Math.round(34 - 26 * tRow)}})`;
  if (v > 0) {{
    const t = maxPos > 0 ? Math.sqrt(Math.min(1, v / maxPos)) : 0;
    return `rgb(${{Math.round(23 - 17 * t)}},${{Math.round(74 + 62 * t)}},${{Math.round(49 - 28 * t)}})`;
  }}
  const t = minNeg < 0 ? Math.sqrt(Math.min(1, v / minNeg)) : 0;
  return `rgb(${{Math.round(88 + 82 * t)}},${{Math.round(34 - 26 * t)}},${{Math.round(34 - 26 * t)}})`;
}}
function drawRewardHeatmap(rs, x, y, width, height) {{
  if (!rewardKeys.length || height <= 0) return;
  ctx.fillStyle = "#d7dee8"; ctx.font = `${{14 * devicePixelRatio}}px Arial`;
  ctx.fillText("reward heatmap", x, y - 8 * devicePixelRatio);
  const headerH = 20 * devicePixelRatio;
  const groupGap = 12 * devicePixelRatio;
  const footerH = 18 * devicePixelRatio;
  const rowH = Math.max(22 * devicePixelRatio, (height - headerH * 2 - groupGap - footerH) / rewardKeys.length);
  const cellW = Math.max(1, width / Math.max(rs.length, 1));
  let minNeg = 0, maxPos = 0;
  const keyStats = rewardKeys.map(k => {{
    let keyMin = 0, keyMax = 0, keyMinMagnitude = Number.POSITIVE_INFINITY, keyMaxMagnitude = 0;
    for (const r of rs) {{
      const v = Number(r[k]) || 0;
      keyMin = Math.min(keyMin, v);
      keyMax = Math.max(keyMax, v);
      const absV = Math.abs(v);
      keyMinMagnitude = Math.min(keyMinMagnitude, absV);
      keyMaxMagnitude = Math.max(keyMaxMagnitude, absV);
    }}
    if (!Number.isFinite(keyMinMagnitude)) keyMinMagnitude = 0;
    return {{k, keyMin, keyMax, keyMinMagnitude, keyMaxMagnitude}};
  }});
  for (const r of rs) for (const k of rewardKeys) {{
    const v = Number(r[k]) || 0;
    minNeg = Math.min(minNeg, v);
    maxPos = Math.max(maxPos, v);
  }}
  const positiveKeys = keyStats.filter(s => rewardSigns[s.k] > 0 || (rewardSigns[s.k] === 0 && s.keyMax >= Math.abs(s.keyMin))).map(s => s.k);
  const negativeKeys = keyStats.filter(s => rewardSigns[s.k] < 0 || (rewardSigns[s.k] === 0 && s.keyMax < Math.abs(s.keyMin))).map(s => s.k);
  let yy = y + 18 * devicePixelRatio;
  yy = drawRewardGroup("positive rewards", positiveKeys, yy);
  yy += groupGap;
  yy = drawRewardGroup("negative rewards", negativeKeys, yy);
  if (hoverRow && hoverRow.sim_time_s >= xMin && hoverRow.sim_time_s <= xMax) {{
    const hx = x + (hoverRow.sim_time_s - xMin) / Math.max(xMax - xMin, 1e-9) * width;
    ctx.strokeStyle = "#e5edf5aa"; ctx.lineWidth = 1 * devicePixelRatio;
    ctx.beginPath(); ctx.moveTo(hx, y + 18 * devicePixelRatio); ctx.lineTo(hx, yy); ctx.stroke();
    drawRewardHoverPanel(hoverRow, x, y + 18 * devicePixelRatio, width);
  }}
  ctx.fillStyle = "#a9b4c2"; ctx.font = `${{12 * devicePixelRatio}}px Arial`;
  ctx.fillText(`red: negative min (${{minNeg.toFixed(4)}}), green: positive max (${{maxPos.toFixed(4)}})`, x, y + height - 4 * devicePixelRatio);

  function drawRewardGroup(title, keys, startY) {{
    ctx.fillStyle = title.startsWith("positive") ? "#7ddc91" : "#ff8d8d";
    ctx.font = `${{13 * devicePixelRatio}}px Arial`;
    ctx.fillText(title, x, startY + 13 * devicePixelRatio);
    let rowY = startY + headerH;
    if (!keys.length) {{
      ctx.fillStyle = "#657184"; ctx.font = `${{12 * devicePixelRatio}}px Arial`;
      ctx.fillText("none", 6 * devicePixelRatio, rowY + rowH * 0.68);
      return rowY + rowH;
    }}
    keys.forEach(k => {{
      const stat = keyStats.find(s => s.k === k);
      const rowMinMagnitude = stat ? stat.keyMinMagnitude : 0;
      const rowMaxMagnitude = stat ? stat.keyMaxMagnitude : 0;
      ctx.fillStyle = "#a9b4c2"; ctx.font = `${{12 * devicePixelRatio}}px Arial`;
      ctx.fillText(k.replace("reward_", ""), 6 * devicePixelRatio, rowY + rowH * 0.68);
      rs.forEach((r, i) => {{
        ctx.fillStyle = heatColor(k, Number(r[k]) || 0, rowMinMagnitude, rowMaxMagnitude, minNeg, maxPos);
        ctx.fillRect(x + i * cellW, rowY, Math.ceil(cellW), Math.max(2 * devicePixelRatio, rowH - 2 * devicePixelRatio));
      }});
      rowY += rowH;
    }});
    ctx.strokeStyle = "#344154";
    ctx.strokeRect(x, startY + headerH, width, rowH * keys.length);
    return rowY;
  }}
}}
function drawRewardHoverPanel(row, x, y, width) {{
  const items = rewardKeys.map(k => {{ return {{k, v: Number(row[k]) || 0}}; }});
  const positiveItems = items.filter(item => rewardSigns[item.k] > 0 || (rewardSigns[item.k] === 0 && item.v >= 0)).sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 8);
  const negativeItems = items.filter(item => rewardSigns[item.k] < 0 || (rewardSigns[item.k] === 0 && item.v < 0)).sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 8);
  if (!positiveItems.length && !negativeItems.length) return;
  const panelW = Math.min(width - 16 * devicePixelRatio, 420 * devicePixelRatio);
  const lineH = 16 * devicePixelRatio;
  const rowCount = positiveItems.length + negativeItems.length + 3;
  const panelH = rowCount * lineH + 12 * devicePixelRatio;
  const px = x + width - panelW - 8 * devicePixelRatio;
  const py = y + 8 * devicePixelRatio;
  const nameX = px + 8 * devicePixelRatio;
  const valueRightX = px + panelW - 8 * devicePixelRatio;
  const valueColW = 86 * devicePixelRatio;
  const nameMaxW = panelW - valueColW - 20 * devicePixelRatio;
  ctx.fillStyle = "rgba(5, 8, 12, 0.86)";
  ctx.strokeStyle = "#3b4656";
  ctx.lineWidth = 1 * devicePixelRatio;
  ctx.beginPath(); ctx.rect(px, py, panelW, panelH); ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#e8edf2"; ctx.font = `${{12 * devicePixelRatio}}px Arial`;
  ctx.fillText(`reward @ t=${{row.sim_time_s.toFixed(3)}}s`, px + 8 * devicePixelRatio, py + 15 * devicePixelRatio);
  let line = 2;
  drawRewardHoverSection("positive", positiveItems, "#7ddc91");
  drawRewardHoverSection("negative", negativeItems, "#ff8d8d");

  function drawRewardHoverSection(title, sectionItems, titleColor) {{
    ctx.fillStyle = titleColor;
    ctx.fillText(title, nameX, py + line * lineH);
    line += 1;
    if (!sectionItems.length) {{
      ctx.fillStyle = "#657184";
      ctx.fillText("none", nameX, py + line * lineH);
      line += 1;
      return;
    }}
    sectionItems.forEach(item => {{
      const yy = py + line * lineH;
      line += 1;
      const name = fitText(item.k.replace("reward_", ""), nameMaxW);
      const value = item.v.toFixed(5);
      ctx.fillStyle = rewardSigns[item.k] < 0 ? "#ff8d8d" : rewardSigns[item.k] > 0 ? "#7ddc91" : (item.v < 0 ? "#ff8d8d" : "#7ddc91");
      ctx.fillText(name, nameX, yy);
      ctx.fillStyle = "#d7dee8";
      ctx.textAlign = "right";
      ctx.fillText(value, valueRightX, yy);
      ctx.textAlign = "left";
    }});
  }}
}}
function fitText(text, maxWidth) {{
  if (ctx.measureText(text).width <= maxWidth) return text;
  if (maxWidth <= ctx.measureText("...").width) return "...";
  let lo = 1, hi = text.length;
  while (lo < hi) {{
    const mid = Math.ceil((lo + hi) / 2);
    const left = Math.ceil(mid / 2);
    const right = Math.floor(mid / 2);
    const candidate = `${{text.slice(0, left)}}...${{text.slice(text.length - right)}}`;
    if (ctx.measureText(candidate).width <= maxWidth) lo = mid;
    else hi = mid - 1;
  }}
  const left = Math.ceil(lo / 2);
  const right = Math.floor(lo / 2);
  return `${{text.slice(0, left)}}...${{text.slice(text.length - right)}}`;
}}
canvas.addEventListener("wheel", e => {{
  if (!e.ctrlKey && !e.metaKey) return;
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const ratio = (e.clientX - rect.left) / rect.width;
  const center = xMin + ratio * (xMax - xMin);
  const factor = e.deltaY < 0 ? 0.85 : 1.18;
  const span = Math.max((xMax - xMin) * factor, 0.05);
  xMin = center - span * ratio; xMax = center + span * (1-ratio);
  draw();
}}, {{passive:false}});
canvas.addEventListener("mousedown", e => {{ dragging=true; lastX=e.clientX; updateHoverFromEvent(e); draw(); }});
canvas.addEventListener("mousemove", e => {{ updateHoverFromEvent(e); if (!dragging) draw(); }});
canvas.addEventListener("mouseleave", () => {{
  if (dragging) return;
  hoverTime = null; hoverRow = null; hoverCanvasX = null; draw();
}});
window.addEventListener("mouseup", () => dragging=false);
window.addEventListener("mousemove", e => {{
  if (!dragging) return;
  const dx = e.clientX - lastX; lastX = e.clientX;
  const span = xMax - xMin;
  const shift = -dx / canvas.clientWidth * span;
  xMin += shift; xMax += shift; updateHoverFromEvent(e); draw();
}});
document.querySelectorAll("input[data-series]").forEach(cb => cb.addEventListener("change", e => {{
  visible[e.target.dataset.series] = e.target.checked; draw();
}}));
document.getElementById("reset").onclick = () => {{ xMin=xFull[0]; xMax=xFull[1]; draw(); }};
window.addEventListener("resize", resize);
resize();
</script>
</body>
</html>
"""
