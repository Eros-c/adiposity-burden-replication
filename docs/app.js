/* Global adiposity-associated disease risk explorer — v3 */
const PALETTE = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"];
const SSP_COLORS = { historical: "#747C80", ssp126: "#1B9E77", ssp245: "#2676A8",
                     ssp370: "#E8871E", ssp585: "#B2182B" };
const SSP_LABELS = { ssp126: "SSP1-2.6", ssp245: "SSP2-4.5", ssp370: "SSP3-7.0", ssp585: "SSP5-8.5" };
const BMI_YEARS = [2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050];
const HIST_LAST = { bmi: 2024, burden: 2021 };   // 历史段结束年（其后为过渡/预测）

let meta = null, countriesLayer = null, overlay = null;
const burdenCache = {};
let selectedIso = null;

const state = {
  tab: "bmi",
  bmi: { metric: "meanbmi", sex: "Both", age: "All", scenario: "ssp245", yearIdx: 7 },
  burden: { disease: "CRC", measure: "deaths", sex: "Both", yearIdx: 6 },
};

/* ---------------- utils ---------------- */
function loadData(src, globalName) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => window[globalName] !== undefined ? resolve(window[globalName])
                                                     : reject(new Error("no data: " + src));
    s.onerror = () => reject(new Error("load failed: " + src));
    document.head.appendChild(s);
  });
}

function fmtNum(v) {
  if (v == null || !isFinite(v)) return "-";
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
  if (a >= 1e3) return (v / 1e3).toFixed(a >= 1e4 ? 0 : 1) + "k";
  if (a >= 100) return Math.round(v).toLocaleString();
  return (Math.round(v * 100) / 100).toLocaleString();
}

function lerpSeries(rec, y0, y1, years) {
  // 从 rec[y0] 到 rec[y1] 线性插值出 years（含端点）的序列
  const v0 = rec[String(y0)], v1 = rec[String(y1)];
  return years.map(y => {
    const t = (y - y0) / (y1 - y0);
    return [y, v0.map((a, i) => a + t * (v1[i] - a))];
  });
}

/* 单调三次样条（Fritsch–Carlson / PCHIP）：斜率连续、不振荡 */
function pchipSlopes(xs, ys) {
  const n = xs.length;
  const m = new Array(n);
  if (n < 2) return m.fill(0);
  const h = [], d = [];
  for (let i = 0; i < n - 1; i++) {
    h.push(xs[i + 1] - xs[i]);
    d.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]));
  }
  m[0] = d[0]; m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) {
    if (d[i - 1] * d[i] <= 0) m[i] = 0;
    else {
      const w1 = 2 * h[i] + h[i - 1], w2 = h[i] + 2 * h[i - 1];
      m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]);
    }
  }
  return m;
}

function pchipEval(xs, ys, m, x) {
  let i = 0;
  while (i < xs.length - 2 && x > xs[i + 1]) i++;
  const h = xs[i + 1] - xs[i], t = (x - xs[i]) / h;
  const t2 = t * t, t3 = t2 * t;
  return (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * m[i]
       + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * m[i + 1];
}

function smoothPoints(xs, ys, years) {
  // 以 xs/ys 为节点，按 PCHIP 求 years 年份的平滑值（含端点）
  const m = pchipSlopes(xs, ys);
  return years.map(x => [x, pchipEval(xs, ys, m, x)]);
}

/* ---------------- data lookup ---------------- */
function bmiRecord() {
  const key = selectedIso || "_GLOBAL";
  const s = state.bmi.sex, a = state.bmi.age;
  const T = window.BMI_TRENDS_V3;
  if (s === "Both" && a === "All") return T.overall[key];
  if (s !== "Both" && a === "All") return (T.sex[s] || {})[key];
  if (s === "Both") return (T.age[a] || {})[key];
  return ((T.sexage[s] || {})[a] || {})[key];
}

async function burdenRecord() {
  const code = state.burden.disease;
  if (!burdenCache[code]) burdenCache[code] = await loadData(`data/trends_burden_${code}.js`, `BURDEN_${code}`);
  return burdenCache[code];
}

function burdenYears() {
  return (meta.burden_map_years_by_code && meta.burden_map_years_by_code[state.burden.disease])
         || [2025, 2050];
}

/* ---------------- series building ---------------- */
function buildSeries(rec, idx, histLast, futureOnlySSP) {
  // 整条曲线（历史+过渡+预测）用同一条 PCHIP 采样后按样式分段：
  // 段间共享节点且斜率连续 → 视觉平滑、无折角
  const out = [];
  const yearsSet = new Set();
  const hist = (rec && rec.historical) || {};
  const hYears = Object.keys(hist).map(Number).filter(y => y <= histLast).sort((a, b) => a - b);
  const sspOrder = futureOnlySSP ? ["ssp245"] : ["ssp126", "ssp245", "ssp370", "ssp585"];
  let histDrawn = false;

  sspOrder.forEach(s => {
    const r = rec && rec[s];
    if (!r) return;
    const pYears = Object.keys(r).map(Number).filter(y => y >= histLast + 1).sort((a, b) => a - b);
    if (!pYears.length) return;
    // 节点 = 历史点 + 预测点（缺年处由 PCHIP 平滑补足）
    const y0 = hYears.length ? hYears[hYears.length - 1] : null;
    const y1raw = pYears[0];
    // 混合区：历史末端 → max(首个预测年, 历史末端+4)——把两套数据源的水平差摊平在 4 年内
    const blendEnd = y0 !== null ? Math.min(Math.max(y1raw, y0 + 4), pYears[pYears.length - 1]) : y1raw;
    const dense = [];
    // 历史段 + 混合区采样（缺口/混合区 0.5 年加密）
    if (y0 !== null) {
      const pkx = [blendEnd, Math.min(blendEnd + 2, pYears[pYears.length - 1])];
      const pky = pkx.map(y => r[String(y)][idx]);
      const kx2 = hYears.concat(pkx);
      const ky2 = hYears.map(y => hist[String(y)][idx]).concat(pky);
      for (let x = hYears[0]; x < y0; x += 1) dense.push(x);
      dense.push(y0);
      for (let x = y0 + 0.5; x < blendEnd; x += 0.5) dense.push(x);
      dense.push(blendEnd);
      const pts2 = smoothPoints(kx2, ky2, dense);
      if (!histDrawn) {
        histDrawn = true;
        out.push({ name: "Historical", kind: "hist",
                   data: pts2.filter(p => p[0] <= y0) });
        out.push({ name: "Transition", kind: "bridge",
                   data: pts2.filter(p => p[0] >= y0 && p[0] <= blendEnd) });
      }
      // 预测段：混合区末端起用真实值
      out.push({ name: SSP_LABELS[s], kind: "proj", color: SSP_COLORS[s],
                 data: pYears.filter(y => y >= blendEnd).map(y => [y, r[String(y)][idx]]) });
      pts2.forEach(p => yearsSet.add(p[0]));
      pYears.filter(y => y >= blendEnd).forEach(y => yearsSet.add(y));
    } else {
      // 无历史段（如负担分性别）：直接预测
      const pts2 = pYears.map(y => [y, r[String(y)][idx]]);
      pts2.forEach(p => yearsSet.add(p[0]));
      out.push({ name: SSP_LABELS[s], kind: "proj", color: SSP_COLORS[s], data: pts2 });
    }
  });

  // 兜底：无预测段时仅画历史
  if (!out.length && hYears.length) {
    const xs = hYears.map(Number);
    const ys = xs.map(y => hist[String(y)][idx]);
    const pts = xs.length >= 2
      ? smoothPoints(xs, ys, range(xs[0], xs[xs.length - 1]))
      : [[xs[0], ys[0]]];
    pts.forEach(p => yearsSet.add(p[0]));
    out.push({ name: "Historical", kind: "hist", data: pts });
  }
  return { years: [...yearsSet].sort((a, b) => a - b), series: out };
}

function range(a, b) {
  const out = [];
  for (let y = a; y <= b; y++) out.push(y);
  return out;
}

/* ---------------- map ---------------- */
function initMap() {
  window.map = L.map("map", {
    minZoom: 2, maxZoom: 7,
    worldCopyJump: false, zoomControl: true, attributionControl: false,
    maxBounds: [[-85, -180], [85, 180]], maxBoundsViscosity: 1.0,
  });
  map.fitBounds([[-58, -170], [82, 185]]);
  loadData("data/countries.js", "COUNTRIES").then(gj => {
    countriesLayer = L.geoJSON(gj, {
      style: { color: "#9AA4AA", weight: 0.4, fill: true, fillColor: "#ffffff",
               fillOpacity: 0.01, interactive: true },
      onEachFeature: (f, layer) => {
        layer.on("mouseover", () => {
          layer.bindTooltip(f.properties.name, { className: "country-tip", sticky: true }).openTooltip();
        });
        layer.on("click", () => { selectedIso = f.properties.iso3; refreshChart(); });
      },
    }).addTo(map);
  });
}

function overlayUrl() {
  if (state.tab === "bmi") {
    const s = state.bmi;
    const year = BMI_YEARS[s.yearIdx];
    const ssp = year === 2050 ? s.scenario : "ssp245";
    return `maps/bmi_${s.metric}_${ssp}_${year}.webp`;
  }
  return `maps/burden_${state.burden.disease}_${state.burden.measure}_ssp245_${burdenYears()[state.burden.yearIdx]}.webp`;
}

function refreshMap() {
  const url = overlayUrl();
  if (overlay) overlay.setUrl(url);
  else overlay = L.imageOverlay(url, [[-85.0511287798066, -180], [85.0511287798066, 180]],
                                { opacity: 1 }).addTo(map);
  if (countriesLayer) countriesLayer.bringToFront();
  refreshLegend();
}

function legendSpec() {
  if (state.tab === "bmi") {
    const cb = meta.colorbar[state.bmi.metric];
    return { label: cb.label, vmin: cb.vmin, vmax: cb.vmax, power: false };
  }
  const key = `${state.burden.disease}_${state.burden.measure}`;
  const vmax = meta.burden_norms[key] || 1;
  const unit = state.burden.measure === "deaths" ? "deaths" : "DALYs";
  return { label: `Attributable ${unit} per 0.25° cell`, vmin: 0, vmax, power: true };
}

function refreshLegend() {
  const spec = legendSpec();
  const ticks = [0, 0.0625, 0.25, 1].map(f => spec.vmin + f * (spec.vmax - spec.vmin));
  const pos = ticks.map(t => {
    const u = (t - spec.vmin) / (spec.vmax - spec.vmin);
    return (spec.power ? Math.sqrt(u) : u) * 100;
  });
  const el = document.getElementById("legend-ticks");
  el.innerHTML = "";
  ticks.forEach((t, i) => {
    const s = document.createElement("span");
    s.style.left = pos[i] + "%";
    s.textContent = fmtNum(Math.round(t * 100) / 100);
    el.appendChild(s);
  });
  document.getElementById("legend-label").textContent = spec.label;
}

/* ---------------- chart ---------------- */
function initChart() {
  window.chart = echarts.init(document.getElementById("chart"), null, { renderer: "canvas" });
  window.chartAge = echarts.init(document.getElementById("chart-age"), null, { renderer: "canvas" });
  window.addEventListener("resize", () => { chart.resize(); chartAge.resize(); });
}

function tooltipFormatter(params) {
  const yv = params.length ? params[0].value[0] : 0;
  const yr = Number.isInteger(yv) ? String(yv) : yv.toFixed(1);
  let html = `<div style="font-weight:600;margin-bottom:2px">${yr}</div>`;
  params.forEach(p => {
    if (p.value[1] == null) return;
    html += `${p.marker} ${p.seriesName}: <b>${fmtNum(p.value[1])}</b><br/>`;
  });
  return html;
}

function renderChart(spec, ylabel) {
  const seriesOut = spec.series.map(s => ({
    name: s.name, type: "line", showSymbol: s.kind === "bridge",
    symbol: "circle", symbolSize: s.kind === "bridge" ? 3.2 : 2.2, smooth: false,
    data: s.data,
    lineStyle: {
      width: s.kind === "hist" ? 2.2 : 1.7,
      type: s.kind === "bridge" ? [6, 4] : "solid",
    },
    color: s.kind === "proj" ? s.color : SSP_COLORS.historical,
    emphasis: { focus: "series" },
    z: s.kind === "proj" ? 3 : 4,
  }));
  chart.setOption({
    animation: false,
    grid: { left: 62, right: 18, top: 34, bottom: 38 },
    legend: { top: 2, right: 8, textStyle: { fontSize: 11 }, itemWidth: 16, itemHeight: 2 },
    xAxis: { type: "value", min: 2015, max: 2050,
             axisLabel: { formatter: v => String(Math.round(v)), fontSize: 11 },
             name: "Year", nameLocation: "middle", nameGap: 24 },
    yAxis: { type: "value", name: ylabel, scale: true,
             axisLabel: { formatter: v => fmtNum(v), fontSize: 11 },
             splitLine: { lineStyle: { color: "#E9ECEE" } } },
    series: seriesOut,
    tooltip: { trigger: "axis", formatter: tooltipFormatter },
  }, true);
}

function renderAgeChart(ageData, ylabel, title) {
  const groups = meta.age_groups;
  const vals = groups.map(g => (ageData && ageData[g]) ? ageData[g][state.burden.measure === "deaths" ? 0 : 1] : 0);
  document.getElementById("chart-age-title").textContent = title;
  chartAge.setOption({
    animation: false,
    grid: { left: 58, right: 12, top: 18, bottom: 30 },
    xAxis: { type: "category", data: groups, axisLabel: { fontSize: 9, rotate: 30 } },
    yAxis: { type: "value", scale: true,
             axisLabel: { formatter: v => fmtNum(v), fontSize: 10 } },
    series: [{ type: "bar", data: vals, itemStyle: { color: "#2676A8" }, barWidth: "62%" }],
    tooltip: { trigger: "axis",
               formatter: p => `${p[0].name}: <b>${fmtNum(p[0].value)}</b>` },
  }, true);
  void ylabel;
}

function chartTitle() {
  const name = selectedIso ? (meta.country_names[selectedIso] || selectedIso) : "Global";
  return name;
}

async function refreshChart() {
  if (state.tab === "bmi") {
    const rec = bmiRecord();
    const idx = state.bmi.metric === "meanbmi" ? 0 : 1;
    const unit = state.bmi.metric === "meanbmi" ? "Mean BMI (kg/m²)" : "Obesity prevalence (%)";
    const sexAge = (state.bmi.sex === "Both" ? "" : ", " + state.bmi.sex)
                 + (state.bmi.age === "All" ? "" : ", age " + state.bmi.age);
    document.getElementById("chart-title").textContent = chartTitle() + " — " + unit + sexAge;
    renderChart(buildSeries(rec, idx, HIST_LAST.bmi, state.bmi.sex !== "Both" || state.bmi.age !== "All"), "");
    document.getElementById("chart-age-wrap").classList.add("hidden");
  } else {
    const trend = await burdenRecord();
    const midx = state.burden.measure === "deaths" ? 0 : 1;
    const unit = state.burden.measure === "deaths" ? "Attributable deaths" : "Attributable DALYs";
    const s = state.burden.sex;
    let rec;
    if (s === "Both") {
      // 仅展示 SSP2-4.5（情景间差异极小，见图注）
      const r245 = selectedIso ? trend.ssp245[selectedIso] : trend._global.ssp245;
      const hist = {}, fut = {};
      Object.keys(r245 || {}).forEach(y => (Number(y) <= HIST_LAST.burden ? hist : fut)[y] = r245[y]);
      rec = { historical: hist, ssp245: fut };
    } else {
      const sr = trend._sex[s] || {};
      rec = {};
      if (sr.ssp245) {
        rec.ssp245 = selectedIso ? (sr.ssp245.countries && sr.ssp245.countries[selectedIso])
                                 : sr.ssp245._global;
      }
    }
    const sexTag = s === "Both" ? "" : ", " + s;
    document.getElementById("chart-title").textContent =
      chartTitle() + " — " + unit + sexTag;
    renderChart(buildSeries(rec, midx, HIST_LAST.burden, false), "");
    // 年龄快照（2050，按性别）
    const wrap = document.getElementById("chart-age-wrap");
    wrap.classList.remove("hidden");
    const snap = (trend._age2050_global || {})[s] || (trend._age2050_global || {}).Both;
    renderAgeChart(snap, unit,
      `Attributable ${state.burden.measure === "deaths" ? "deaths" : "DALYs"} by age group, 2050 (SSP2-4.5, ${s.toLowerCase()})`);
  }
}

async function refreshAll() {
  refreshMap();
  await refreshChart();
}

/* ---------------- controls ---------------- */
function initControls() {
  const sspSel = [document.getElementById("bmi-scenario")];
  sspSel.forEach(sel => {
    meta.scenarios.forEach(s => {
      const o = document.createElement("option");
      o.value = s.code; o.textContent = s.label;
      if (s.code === "ssp245") o.selected = true;
      sel.appendChild(o);
    });
  });
  const ageSel = document.getElementById("bmi-age");
  ["All ages 25+"].concat(meta.age_groups).forEach(a => {
    const o = document.createElement("option");
    o.value = a === "All ages 25+" ? "All" : a;
    o.textContent = a;
    ageSel.appendChild(o);
  });
  const dSel = document.getElementById("burden-disease");
  meta.diseases.forEach(d => {
    const o = document.createElement("option");
    o.value = d.code; o.textContent = d.name;
    dSel.appendChild(o);
  });
  dSel.value = state.burden.disease;
  {
    const yrs = burdenYears();
    const slider = document.getElementById("burden-year");
    slider.max = String(yrs.length - 1);
    state.burden.yearIdx = yrs.length - 1;
    slider.value = String(state.burden.yearIdx);
    document.getElementById("burden-year-label").textContent = yrs[state.burden.yearIdx];
  }
  document.getElementById("bmi-year-label").textContent = BMI_YEARS[state.bmi.yearIdx];

  document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    state.tab = t.dataset.tab;
    document.getElementById("controls-bmi").classList.toggle("hidden", state.tab !== "bmi");
    document.getElementById("controls-burden").classList.toggle("hidden", state.tab !== "burden");
    refreshAll();
  }));
  document.getElementById("bmi-metric").addEventListener("change", e => {
    state.bmi.metric = e.target.value; refreshAll();
  });
  document.getElementById("bmi-sex").addEventListener("change", e => {
    state.bmi.sex = e.target.value; refreshChart();
  });
  document.getElementById("bmi-age").addEventListener("change", e => {
    state.bmi.age = e.target.value; refreshChart();
  });
  document.getElementById("bmi-scenario").addEventListener("change", e => {
    state.bmi.scenario = e.target.value; refreshAll();
  });
  document.getElementById("bmi-year").addEventListener("input", e => {
    state.bmi.yearIdx = +e.target.value;
    document.getElementById("bmi-year-label").textContent = BMI_YEARS[state.bmi.yearIdx];
    refreshMap();
  });
  document.getElementById("burden-disease").addEventListener("change", e => {
    state.burden.disease = e.target.value;
    const yrs = burdenYears();
    const slider = document.getElementById("burden-year");
    slider.max = String(yrs.length - 1);
    state.burden.yearIdx = yrs.length - 1;
    slider.value = String(state.burden.yearIdx);
    document.getElementById("burden-year-label").textContent = yrs[state.burden.yearIdx];
    refreshAll();
  });
  document.getElementById("burden-measure").addEventListener("change", e => {
    state.burden.measure = e.target.value; refreshAll();
  });
  document.getElementById("burden-sex").addEventListener("change", e => {
    state.burden.sex = e.target.value; refreshChart();
  });
  document.getElementById("burden-year").addEventListener("input", e => {
    state.burden.yearIdx = +e.target.value;
    document.getElementById("burden-year-label").textContent = burdenYears()[state.burden.yearIdx];
    refreshMap();
  });
  document.getElementById("reset-country").addEventListener("click", () => {
    selectedIso = null; refreshChart();
  });
}

async function init() {
  meta = await loadData("data/meta.js", "META");
  await loadData("data/trends_bmi_v3.js", "BMI_TRENDS_V3");
  initMap();
  initControls();
  initChart();
  await refreshAll();
}

init().catch(e => { document.body.innerHTML = "<pre style='padding:40px'>" + e.stack + "</pre>"; });
