/* Global adiposity-associated disease risk explorer — main logic */

const PALETTE = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"];
const SSP_COLORS = { historical: "#747C80", ssp126: "#1B9E77", ssp245: "#2676A8",
                     ssp370: "#E8871E", ssp585: "#B2182B" };
const SSP_LABELS = { ssp126: "SSP1-2.6", ssp245: "SSP2-4.5", ssp370: "SSP3-7.0", ssp585: "SSP5-8.5" };
const BMI_YEARS = [2015, 2020, 2024, 2025, 2035, 2050];
const BURDEN_YEARS = [2015, 2025, 2050];

let meta = null, countriesLayer = null, overlay = null;
let bmiTrends = null, bmiGlobal = null;
const burdenCache = {};
let selectedIso = null;   // null = global

const state = {
  tab: "bmi",
  bmi: { metric: "meanbmi", scenario: "ssp245", yearIdx: 5 },
  burden: { disease: "CRC", measure: "deaths", scenario: "ssp245", yearIdx: 2 },
};

/* ---------------- data loading ---------------- */
async function loadJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error("load failed: " + path);
  return r.json();
}

async function init() {
  meta = await loadJSON("data/meta.json");
  bmiTrends = await loadJSON("data/trends_bmi_national.json");
  bmiGlobal = await loadJSON("data/trends_bmi_global.json");
  initMap();
  initControls();
  initChart();
  await refreshAll();
}

/* ---------------- map ---------------- */
function initMap() {
  window.map = L.map("map", {
    minZoom: 2, maxZoom: 7,
    worldCopyJump: false, zoomControl: true, attributionControl: false,
    maxBounds: [[-85, -180], [85, 180]], maxBoundsViscosity: 1.0,
  });
  map.fitBounds([[-58, -170], [82, 185]]);
  // 白底，无瓦片；仅国界
  fetch("data/countries.geojson").then(r => r.json()).then(gj => {
    countriesLayer = L.geoJSON(gj, {
      style: { color: "#9AA4AA", weight: 0.4, fill: true, fillColor: "#ffffff",
               fillOpacity: 0.01, interactive: true },
      onEachFeature: (f, layer) => {
        layer.on("mouseover", () => {
          layer.bindTooltip(f.properties.name, { className: "country-tip", sticky: true }).openTooltip();
        });
        layer.on("click", () => {
          selectedIso = f.properties.iso3;
          refreshChart();
        });
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
  const s = state.burden;
  const year = BURDEN_YEARS[s.yearIdx];
  const ssp = year === 2050 ? s.scenario : "ssp245";
  return `maps/burden_${s.disease}_${s.measure}_${ssp}_${year}.webp`;
}

function refreshMap() {
  const url = overlayUrl();
  if (overlay) overlay.setUrl(url);
  else overlay = L.imageOverlay(url, [[-85, -180], [85, 180]], { opacity: 1 }).addTo(map);
  if (countriesLayer) countriesLayer.bringToFront();
  refreshLegend();
}

/* ---------------- legend ---------------- */
function legendSpec() {
  if (state.tab === "bmi") {
    const m = state.bmi.metric;
    const cb = meta.colorbar[m];
    return { label: cb.label, vmin: cb.vmin, vmax: cb.vmax };
  }
  const key = `${state.burden.disease}_${state.burden.measure}`;
  const vmax = meta.burden_norms[key] || 1;
  const unit = state.burden.measure === "deaths" ? "deaths" : "DALYs";
  return { label: `Attributable ${unit} per 0.25° cell`, vmin: 0, vmax };
}

function refreshLegend() {
  const spec = legendSpec();
  const isBurden = state.tab === "burden";   // 负担图导出时用 PowerNorm(0.5)
  const ticks = [0, 0.0625, 0.25, 1].map(f => spec.vmin + f * (spec.vmax - spec.vmin));
  const pos = ticks.map(t => {
    const u = (t - spec.vmin) / (spec.vmax - spec.vmin);
    return (isBurden ? Math.sqrt(u) : u) * 100;
  });
  const el = document.getElementById("legend-ticks");
  el.innerHTML = "";
  ticks.forEach((t, i) => {
    const s = document.createElement("span");
    s.style.left = pos[i] + "%";
    s.textContent = t >= 1000 ? (t / 1000).toFixed(1) + "k" : (Math.round(t * 10) / 10);
    el.appendChild(s);
  });
  document.getElementById("legend-label").textContent = spec.label;
}

/* ---------------- controls ---------------- */
function initControls() {
  const sspSel = [document.getElementById("bmi-scenario"),
                  document.getElementById("burden-scenario")];
  sspSel.forEach(sel => {
    meta.scenarios.forEach(s => {
      const o = document.createElement("option");
      o.value = s.code; o.textContent = s.label;
      if (s.code === "ssp245") o.selected = true;
      sel.appendChild(o);
    });
  });
  const dSel = document.getElementById("burden-disease");
  meta.diseases.forEach(d => {
    const o = document.createElement("option");
    o.value = d.code; o.textContent = d.name;
    dSel.appendChild(o);
  });

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
  document.getElementById("bmi-scenario").addEventListener("change", e => {
    state.bmi.scenario = e.target.value; refreshAll();
  });
  document.getElementById("bmi-year").addEventListener("input", e => {
    state.bmi.yearIdx = +e.target.value;
    document.getElementById("bmi-year-label").textContent = BMI_YEARS[state.bmi.yearIdx];
    refreshAll();
  });
  document.getElementById("burden-disease").addEventListener("change", e => {
    state.burden.disease = e.target.value; refreshAll();
  });
  document.getElementById("burden-measure").addEventListener("change", e => {
    state.burden.measure = e.target.value; refreshAll();
  });
  document.getElementById("burden-scenario").addEventListener("change", e => {
    state.burden.scenario = e.target.value; refreshAll();
  });
  document.getElementById("burden-year").addEventListener("input", e => {
    state.burden.yearIdx = +e.target.value;
    document.getElementById("burden-year-label").textContent = BURDEN_YEARS[state.burden.yearIdx];
    refreshAll();
  });
  document.getElementById("reset-country").addEventListener("click", () => {
    selectedIso = null; refreshChart();
  });
}

/* ---------------- chart ---------------- */
function initChart() {
  window.chart = echarts.init(document.getElementById("chart"), null, { renderer: "canvas" });
  window.addEventListener("resize", () => chart.resize());
}

function chartTitle() {
  const name = selectedIso ? (meta.country_names[selectedIso] || selectedIso) : "Global";
  return name;
}

function bmiSeries() {
  // 返回 {years[], series: [{name, data[]}]}
  const metricIdx = state.bmi.metric === "meanbmi" ? 0 : 2;
  const src = selectedIso ? bmiTrends[selectedIso] : null;
  const glob = bmiGlobal;
  const series = [];
  const yearsSet = new Set();
  const histSrc = selectedIso ? (src && src.historical) : glob.historical;
  if (histSrc) {
    const yrs = Object.keys(histSrc).map(Number).sort((a, b) => a - b);
    yrs.forEach(y => yearsSet.add(y));
    series.push({ name: "Historical", years: yrs,
                  data: yrs.map(y => histSrc[y][metricIdx]) });
  }
  meta.scenarios.forEach(s => {
    const rec = selectedIso ? (src && src[s.code]) : glob[s.code];
    if (!rec) return;
    const yrs = Object.keys(rec).map(Number).sort((a, b) => a - b);
    yrs.forEach(y => yearsSet.add(y));
    series.push({ name: SSP_LABELS[s.code], years: yrs,
                  data: yrs.map(y => rec[y][metricIdx]) });
  });
  return { years: [...yearsSet].sort((a, b) => a - b), series };
}

function burdenSeries(trend) {
  const midx = state.burden.measure === "deaths" ? 0 : 1;
  const series = [];
  const yearsSet = new Set();
  meta.scenarios.forEach(s => {
    let rec2 = selectedIso
      ? (trend[s.code] && trend[s.code][selectedIso])
      : trend._global[s.code];
    if (!rec2) return;
    const yrs = Object.keys(rec2).map(Number).sort((a, b) => a - b);
    yrs.forEach(y => yearsSet.add(y));
    series.push({ name: SSP_LABELS[s.code], years: yrs,
                  data: yrs.map(y => rec2[y][midx] / 1e6) });
  });
  return { years: [...yearsSet].sort((a, b) => a - b), series };
}

function renderChart(spec, ylabel) {
  const hist = spec.series.filter(s => s.name === "Historical");
  const proj = spec.series.filter(s => s.name !== "Historical");
  const seriesOut = [];
  // 历史段 + 每个情景的线段（2025 起）
  const sspLine = (s, isHist) => ({
    name: s.name, type: "line", showSymbol: false, symbol: "none",
    data: s.years.map((y, i) => [y, s.data[i]]),
    lineStyle: { width: isHist ? 2.2 : 1.6, type: isHist ? "solid" : "solid" },
    color: isHist ? SSP_COLORS.historical : undefined,
    emphasis: { focus: "series" },
  });
  hist.forEach(s => seriesOut.push(sspLine(s, true)));
  proj.forEach(s => seriesOut.push(sspLine(s, false)));
  if (seriesOut.length) {
    seriesOut[0].markLine = {
      symbol: "none", silent: true, label: { show: false },
      lineStyle: { color: "#B0B7BC", type: "dashed", width: 1 },
      data: [{ xAxis: state.tab === "bmi" ? 2024.5 : 2021.5 }],
    };
  }
  chart.setOption({
    animation: false,
    grid: { left: 64, right: 18, top: 34, bottom: 40 },
    legend: { top: 2, right: 8, textStyle: { fontSize: 11 }, itemWidth: 16, itemHeight: 2 },
    xAxis: { type: "value", min: 2015, max: 2050,
             axisLabel: { formatter: v => v, fontSize: 11 }, name: "Year",
             nameLocation: "middle", nameGap: 26 },
    yAxis: { type: "value", name: ylabel, scale: true,
             axisLabel: { fontSize: 11 },
             splitLine: { lineStyle: { color: "#E9ECEE" } } },
    series: seriesOut,
    color: meta.scenarios.map(s => SSP_COLORS[s.code]),
    tooltip: { trigger: "axis", valueFormatter: v => v == null ? "-" : Number(v).toLocaleString() },
  }, true);
}

async function refreshChart() {
  if (state.tab === "bmi") {
    const spec = bmiSeries();
    const unit = state.bmi.metric === "meanbmi" ? "Mean BMI (kg/m²)" : "Obesity prevalence (%)";
    document.getElementById("chart-title").textContent = chartTitle() + " — " + unit;
    renderChart(spec, "");
  } else {
    const code = state.burden.disease;
    if (!burdenCache[code]) burdenCache[code] = await loadJSON(`data/trends_burden_${code}.json`);
    const spec = burdenSeries(burdenCache[code]);
    const unit = state.burden.measure === "deaths" ? "Attributable deaths (millions)"
                                                   : "Attributable DALYs (millions)";
    document.getElementById("chart-title").textContent = chartTitle() + " — " + unit;
    renderChart(spec, "");
  }
}

async function refreshAll() {
  refreshMap();
  await refreshChart();
}

init().catch(e => { document.body.innerHTML = "<pre style='padding:40px'>" + e.stack + "</pre>"; });
