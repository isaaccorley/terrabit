import * as duckdb from "@duckdb/duckdb-wasm";
import duckdbWasmEh from "@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url";
import duckdbWasmMvp from "@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url";
import duckdbWorkerEh from "@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url";
import duckdbWorkerMvp from "@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url";

import "./styles.css";
import { GlobeMap } from "./map";
import type {
  BBox,
  CandidateRow,
  ManifestRow,
  PositiveMatch,
  PositivePoint,
  RankedRow,
  ViewMode,
} from "./types";
import {
  centroid,
  containsPoint,
  distanceSquared,
  formatLatLng,
  normalizeBBox,
  normalizeEmbedding,
} from "./util";

const MANIFEST_URL =
  "https://data.source.coop/geovibes/terrabit/clay-v1_5-binary-sentinel-2/manifest.parquet";
const DEFAULT_TOP_K = 10;
const MAX_TOP_K = 50;

type AppState = {
  bbox: BBox | null;
  status: string;
  manifestShards: ManifestRow[];
  candidateRows: CandidateRow[];
  positivePoints: PositivePoint[];
  positiveMatches: PositiveMatch[];
  results: RankedRow[];
  outlierResults: RankedRow[];
  outlierComputed: boolean;
  topK: number;
  viewMode: ViewMode;
  threshold: number;
  loading: boolean;
};

const state: AppState = {
  bbox: null,
  status: "Spin the globe. Zoom. Shift-drag or hit Draw region to define an AOI.",
  manifestShards: [],
  candidateRows: [],
  positivePoints: [],
  positiveMatches: [],
  results: [],
  outlierResults: [],
  outlierComputed: false,
  topK: DEFAULT_TOP_K,
  viewMode: "topk",
  threshold: Infinity,
  loading: false,
};

let globe: GlobeMap;
let dbPromise: Promise<duckdb.AsyncDuckDB> | null = null;
let scoringWorker: Worker | null = null;
let scoringWorkerReady = false;
let scoringRequestId = 0;
let latestScoreRunId = 0;
let latestLoadRunId = 0;

const DUCKDB_BUNDLES: duckdb.DuckDBBundles = {
  mvp: { mainModule: duckdbWasmMvp, mainWorker: duckdbWorkerMvp },
  eh: { mainModule: duckdbWasmEh, mainWorker: duckdbWorkerEh },
};

/* ------------------------------------------------------------------ UI shell */

function renderShell(): void {
  const app = document.querySelector<HTMLDivElement>("#app");
  if (!app) throw new Error("App root missing");
  document.title = "terrabit — binary earth embedding retrieval";
  app.innerHTML = `
    <div class="viewport">
      <div id="map" class="map-canvas"></div>

      <header class="hud-brand">
        <div class="brand-mark" aria-hidden="true">
          <span class="brand-ring"></span>
          <span class="brand-ring brand-ring-2"></span>
          <span class="brand-dot"></span>
        </div>
        <div class="brand-text">
          <h1>terrabit</h1>
          <p>binary earth embedding retrieval</p>
        </div>
      </header>

      <div class="hud-search" id="search-wrap">
        <div class="search-bar">
          <svg class="search-glyph" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
            <circle cx="11" cy="11" r="7" />
            <path d="m20 20-3.5-3.5" />
          </svg>
          <input
            id="search-input"
            type="text"
            autocomplete="off"
            spellcheck="false"
            placeholder="Search a place — country, city, park, coordinates…"
          />
          <span id="search-spinner" class="search-spinner" aria-hidden="true"></span>
          <kbd class="search-kbd">/</kbd>
        </div>
        <ul id="search-results" class="search-results" role="listbox"></ul>
      </div>

      <div class="hud-status" id="status-pill">
        <span class="status-led"></span>
        <span id="status-text">${state.status}</span>
      </div>

      <section class="hud-panel hud-panel-left" id="query-panel">
        <header class="panel-head">
          <span class="panel-kicker">01 · Query</span>
        </header>
        <div class="draw-row">
          <button id="draw-btn" class="btn btn-sm btn-primary" type="button">
            <span class="btn-glyph">▢</span>
            <span id="draw-label">Draw region</span>
          </button>
          <button id="clear-region-btn" class="btn btn-sm btn-ghost" type="button">Clear region</button>
          <button id="clear-points-btn2" class="btn btn-sm btn-ghost" type="button">Clear points</button>
        </div>
        <div class="meta-row">
          <div class="meta-cell">
            <dt>Shards</dt>
            <dd id="m-shards">—</dd>
          </div>
          <div class="meta-cell">
            <dt>Patches</dt>
            <dd id="m-patches">—</dd>
          </div>
          <div class="meta-cell">
            <dt>ROI</dt>
            <dd id="m-roi">—</dd>
          </div>
        </div>

        <div class="panel-split">
          <div class="sub-card">
            <header class="sub-head">
              <div>
                <span class="panel-kicker">02 · Exemplars</span>
              </div>
              <div class="sub-head-actions">
                <span id="exemplar-count" class="count-badge">0</span>
                <button id="clear-points-btn" class="icon-btn" type="button" title="Clear points" aria-label="Clear points">
                  <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>
                </button>
              </div>
            </header>
            <ol id="positive-list" class="exemplar-list"></ol>
          </div>

          <div class="sub-card sub-card-retrieval">
            <header class="sub-head sub-head-stack">
              <div>
                <span class="panel-kicker">03 · Retrieval</span>

              </div>
              <span id="result-count" class="result-summary"></span>
            </header>

            <div class="view-toggle" role="tablist" aria-label="Result view">
              <button data-view="topk" class="view-tab is-active" type="button" role="tab" title="Top-K" aria-label="Top-K ranked list">
                <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="2" y="3" width="12" height="3" rx="0.6"/><rect x="2" y="7" width="9" height="3" rx="0.6"/><rect x="2" y="11" width="5" height="3" rx="0.6"/></svg>
              </button>
              <button data-view="heatmap" class="view-tab" type="button" role="tab" title="Heatmap" aria-label="Heatmap of all scored tiles">
                <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="2" y="2" width="4" height="4"/><rect x="6" y="2" width="4" height="4"/><rect x="10" y="2" width="4" height="4"/><rect x="2" y="6" width="4" height="4"/><rect x="6" y="6" width="4" height="4"/><rect x="10" y="6" width="4" height="4"/><rect x="2" y="10" width="4" height="4"/><rect x="6" y="10" width="4" height="4"/><rect x="10" y="10" width="4" height="4"/></svg>
              </button>
              <button data-view="contour" class="view-tab" type="button" role="tab" title="Contour" aria-label="Smooth density surface">
                <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M2 12c2-4 4-6 6-7s4 0 6 3"/><path d="M2 10c2-3 4-4 6-5s4 0 6 2" opacity=".5"/></svg>
              </button>
              <button data-view="outlier" class="view-tab" type="button" role="tab" title="Outlier" aria-label="Most unique patches">
                <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="8" cy="8" r="2"/><circle cx="4" cy="6" r="1.2"/><circle cx="12" cy="5" r="1.2"/><circle cx="13" cy="11" r="1.2"/><circle cx="3" cy="12" r="1.2"/></svg>
              </button>
              <button data-view="threshold" class="view-tab" type="button" role="tab" title="Cutoff" aria-label="Distance cutoff">
                <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="2" y="10" width="2" height="4"/><rect x="5" y="6" width="2" height="8"/><rect x="8" y="3" width="2" height="11"/><rect x="11" y="8" width="2" height="6"/><line x1="1" y1="7" x2="15" y2="7" stroke-dasharray="2 1.5"/></svg>
              </button>
            </div>

            <div id="topk-control" class="slider-row" hidden>
              <label class="slider">
                <span>Top-k <strong id="topk-value">${DEFAULT_TOP_K}</strong></span>
                <input id="topk-slider" type="range" min="1" max="${MAX_TOP_K}" value="${DEFAULT_TOP_K}" />
              </label>
            </div>

            <div id="heatmap-legend" class="heatmap-legend" hidden>
              <div class="legend-gradient"></div>
              <div class="legend-labels">
                <span>Similar</span>
                <span>Distant</span>
              </div>
            </div>

            <div id="outlier-legend" class="heatmap-legend" hidden>
              <div class="legend-gradient legend-gradient-outlier"></div>
              <div class="legend-labels">
                <span>Most unique</span>
                <span>Common</span>
              </div>
            </div>

            <div id="threshold-control" class="threshold-control" hidden>
              <div id="histogram-wrap" class="histogram-wrap"></div>
              <label class="slider">
                <span>Distance ≤ <strong id="threshold-value">0</strong> · <strong id="threshold-count">0</strong> patches</span>
                <input id="threshold-slider" type="range" min="0" max="100" value="50" step="0.1" />
              </label>
            </div>

            <ol id="result-list" class="result-list"></ol>
            <button id="export-btn" class="btn btn-sm btn-ghost btn-export" type="button" hidden>
              <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M2 11v3h12v-3"/><path d="M8 2v8"/><path d="M5 7l3 3 3-3"/></svg>
              <span>Export GeoParquet</span>
            </button>
          </div>
        </div>
      </section>

      <nav class="hud-panel hud-panel-right hud-aoi-nav" id="aoi-nav">
        <header class="panel-head">
          <span class="panel-kicker">Explore</span>
        </header>
        <ul id="aoi-list" class="aoi-list"></ul>
      </nav>
    </div>
  `;
}

function els() {
  return {
    status: document.querySelector<HTMLElement>("#status-text"),
    statusPill: document.querySelector<HTMLElement>("#status-pill"),
    drawBtn: document.querySelector<HTMLButtonElement>("#draw-btn"),
    drawLabel: document.querySelector<HTMLElement>("#draw-label"),
    clearRegionBtn: document.querySelector<HTMLButtonElement>("#clear-region-btn"),
    clearPointsBtn2: document.querySelector<HTMLButtonElement>("#clear-points-btn2"),
    mShards: document.querySelector<HTMLElement>("#m-shards"),
    mPatches: document.querySelector<HTMLElement>("#m-patches"),
    mRoi: document.querySelector<HTMLElement>("#m-roi"),
    positiveList: document.querySelector<HTMLOListElement>("#positive-list"),
    exemplarCount: document.querySelector<HTMLElement>("#exemplar-count"),
    topkSlider: document.querySelector<HTMLInputElement>("#topk-slider"),
    topkValue: document.querySelector<HTMLElement>("#topk-value"),
    topkControl: document.querySelector<HTMLElement>("#topk-control"),
    viewTabs: document.querySelectorAll<HTMLButtonElement>(".view-tab[data-view]"),
    heatmapLegend: document.querySelector<HTMLElement>("#heatmap-legend"),
    outlierLegend: document.querySelector<HTMLElement>("#outlier-legend"),
    thresholdControl: document.querySelector<HTMLElement>("#threshold-control"),
    thresholdSlider: document.querySelector<HTMLInputElement>("#threshold-slider"),
    thresholdValue: document.querySelector<HTMLElement>("#threshold-value"),
    thresholdCount: document.querySelector<HTMLElement>("#threshold-count"),
    histogramWrap: document.querySelector<HTMLElement>("#histogram-wrap"),
    clearPointsBtn: document.querySelector<HTMLButtonElement>("#clear-points-btn"),
    resultCount: document.querySelector<HTMLElement>("#result-count"),
    resultList: document.querySelector<HTMLOListElement>("#result-list"),
    exportBtn: document.querySelector<HTMLButtonElement>("#export-btn"),
    searchWrap: document.querySelector<HTMLElement>("#search-wrap"),
    searchInput: document.querySelector<HTMLInputElement>("#search-input"),
    searchResults: document.querySelector<HTMLUListElement>("#search-results"),
    searchSpinner: document.querySelector<HTMLElement>("#search-spinner"),
    aoiList: document.querySelector<HTMLUListElement>("#aoi-list"),
  };
}

/* --------------------------------------------------------------- AOI presets */

type AoiPreset = { name: string; tag: string; bbox: BBox };

const AOI_PRESETS: AoiPreset[] = [
  { name: "Center pivots", tag: "agriculture", bbox: { west: 37.2, south: 29.0, east: 39.0, north: 30.2 } },
  { name: "Palm Islands", tag: "coastal eng.", bbox: { west: 54.95, south: 25.05, east: 55.25, north: 25.2 } },
  { name: "Bhadla Solar", tag: "solar farm", bbox: { west: 71.6, south: 27.3, east: 72.1, north: 27.65 } },
  { name: "Rondônia", tag: "deforestation", bbox: { west: -63.5, south: -11.0, east: -62.5, north: -10.0 } },
  { name: "Kansas Grid", tag: "cropland", bbox: { west: -101.0, south: 37.6, east: -100.0, north: 38.4 } },
  { name: "Rotterdam", tag: "port", bbox: { west: 3.85, south: 51.85, east: 4.35, north: 52.05 } },
  { name: "Hartsfield ATL", tag: "airport", bbox: { west: -84.55, south: 33.55, east: -84.35, north: 33.7 } },
  { name: "Pilbara Mines", tag: "mining", bbox: { west: 117.6, south: -22.8, east: 118.8, north: -21.8 } },
  { name: "Venice Lagoon", tag: "lagoon", bbox: { west: 12.15, south: 45.3, east: 12.55, north: 45.55 } },
  { name: "Vatnajökull", tag: "glacier", bbox: { west: -17.2, south: 64.1, east: -16.2, north: 64.6 } },
  { name: "Ganges Delta", tag: "river delta", bbox: { west: 89.0, south: 21.6, east: 90.0, north: 22.4 } },
  { name: "Malé Atoll", tag: "coral atoll", bbox: { west: 73.35, south: 4.05, east: 73.7, north: 4.35 } },
  { name: "Yellowstone", tag: "caldera", bbox: { west: -111.0, south: 44.3, east: -110.0, north: 44.85 } },
  { name: "Mekong Delta", tag: "rice paddy", bbox: { west: 105.6, south: 9.7, east: 106.4, north: 10.3 } },
  { name: "Horns Rev", tag: "wind farm", bbox: { west: 7.5, south: 55.4, east: 8.2, north: 55.8 } },
  { name: "Salar de Uyuni", tag: "salt flat", bbox: { west: -68.0, south: -20.6, east: -67.0, north: -19.8 } },
  { name: "Tokyo Bay", tag: "megacity", bbox: { west: 139.6, south: 35.55, east: 140.0, north: 35.8 } },
  { name: "Outer Banks", tag: "barrier island", bbox: { west: -76.0, south: 35.0, east: -75.3, north: 35.7 } },
  { name: "Bali Terraces", tag: "terraced ag.", bbox: { west: 115.15, south: -8.55, east: 115.55, north: -8.2 } },
  { name: "Borneo Palm", tag: "oil palm", bbox: { west: 109.5, south: 0.6, east: 110.5, north: 1.4 } },
  { name: "Nile Valley", tag: "irrigated strip", bbox: { west: 31.0, south: 25.5, east: 31.8, north: 26.3 } },
  { name: "Las Vegas", tag: "desert city", bbox: { west: -115.35, south: 36.0, east: -114.95, north: 36.3 } },
  { name: "Namib Dunes", tag: "sand dunes", bbox: { west: 14.8, south: -24.8, east: 15.6, north: -24.0 } },
  { name: "Three Gorges", tag: "dam/reservoir", bbox: { west: 110.8, south: 30.7, east: 111.4, north: 31.1 } },
  { name: "Atacama", tag: "lithium mines", bbox: { west: -68.6, south: -23.8, east: -67.8, north: -23.1 } },
  { name: "Svalbard", tag: "arctic coast", bbox: { west: 14.5, south: 78.0, east: 16.5, north: 78.5 } },
  { name: "Aral Sea", tag: "dried lake", bbox: { west: 58.0, south: 44.5, east: 59.5, north: 45.5 } },
  { name: "Great Reef", tag: "coral reef", bbox: { west: 145.6, south: -16.8, east: 146.4, north: -16.2 } },
  { name: "Brasília", tag: "planned city", bbox: { west: -48.0, south: -15.9, east: -47.7, north: -15.65 } },
  { name: "Suez Canal", tag: "shipping canal", bbox: { west: 32.2, south: 30.3, east: 32.6, north: 31.0 } },
  { name: "Iceland Lava", tag: "lava field", bbox: { west: -22.5, south: 63.7, east: -21.5, north: 64.1 } },
  { name: "Saharan Oasis", tag: "oasis", bbox: { west: 8.8, south: 32.3, east: 9.4, north: 32.7 } },
  { name: "Amsterdam", tag: "canal city", bbox: { west: 4.8, south: 52.33, east: 5.0, north: 52.42 } },
  { name: "Everglades", tag: "wetland", bbox: { west: -81.0, south: 25.3, east: -80.3, north: 25.8 } },
  { name: "Danakil", tag: "salt/sulfur", bbox: { west: 40.2, south: 14.1, east: 40.7, north: 14.5 } },
  { name: "Singapore", tag: "island city", bbox: { west: 103.6, south: 1.2, east: 104.05, north: 1.45 } },
  { name: "Fjords Norway", tag: "fjord", bbox: { west: 6.5, south: 61.5, east: 7.5, north: 62.0 } },
  { name: "Mount Etna", tag: "volcano", bbox: { west: 14.85, south: 37.65, east: 15.15, north: 37.85 } },
  { name: "Cape Town", tag: "coastal city", bbox: { west: 18.3, south: -34.1, east: 18.7, north: -33.85 } },
];

function renderAoiPresets(): void {
  const e = els();
  if (!e.aoiList) return;
  e.aoiList.innerHTML = AOI_PRESETS.map(
    (aoi, i) => `
    <li class="aoi-item" style="--i:${i}">
      <button type="button" data-aoi="${i}">
        <span class="aoi-name">${aoi.name}</span>
        <span class="aoi-tag">${aoi.tag}</span>
      </button>
    </li>`,
  ).join("");
  e.aoiList.querySelectorAll<HTMLButtonElement>("button[data-aoi]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const aoi = AOI_PRESETS[Number(btn.dataset.aoi)];
      globe.fitBounds(aoi.bbox, { padding: 60, maxZoom: 11 });
      setStatus(`${aoi.name} — shift-drag to draw a region.`);
    });
  });
}

/* --------------------------------------------------------------- Geocoding */

type GeocodeHit = {
  label: string;
  sublabel: string;
  lat: number;
  lng: number;
  bbox?: BBox;
  type: string;
};

let geocodeReqId = 0;
let geocodeTimer: number | null = null;

async function fetchGeocode(query: string): Promise<GeocodeHit[]> {
  const url = new URL("https://nominatim.openstreetmap.org/search");
  url.searchParams.set("q", query);
  url.searchParams.set("format", "jsonv2");
  url.searchParams.set("limit", "6");
  url.searchParams.set("addressdetails", "0");
  const res = await fetch(url.toString(), {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) return [];
  const rows = (await res.json()) as Array<{
    display_name: string;
    lat: string;
    lon: string;
    type: string;
    class: string;
    boundingbox?: [string, string, string, string];
  }>;
  return rows.map((r) => {
    const parts = r.display_name.split(",").map((s) => s.trim());
    const label = parts[0] ?? r.display_name;
    const sublabel = parts.slice(1, 4).join(" · ");
    const bb = r.boundingbox
      ? {
          south: Number(r.boundingbox[0]),
          north: Number(r.boundingbox[1]),
          west: Number(r.boundingbox[2]),
          east: Number(r.boundingbox[3]),
        }
      : undefined;
    return {
      label,
      sublabel,
      lat: Number(r.lat),
      lng: Number(r.lon),
      bbox: bb,
      type: `${r.class}:${r.type}`,
    };
  });
}

function tryParseLatLng(query: string): GeocodeHit | null {
  const m = query.trim().match(/^\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*$/);
  if (!m) return null;
  const lat = Number(m[1]);
  const lng = Number(m[2]);
  if (!isFinite(lat) || !isFinite(lng)) return null;
  if (Math.abs(lat) > 90 || Math.abs(lng) > 180) return null;
  return {
    label: `${lat.toFixed(4)}, ${lng.toFixed(4)}`,
    sublabel: "coordinate",
    lat,
    lng,
    type: "coord",
  };
}

function renderGeocodeResults(hits: GeocodeHit[], query: string): void {
  const e = els();
  if (!e.searchResults || !e.searchWrap) return;
  if (!query) {
    e.searchWrap.classList.remove("is-open");
    e.searchResults.innerHTML = "";
    return;
  }
  if (!hits.length) {
    e.searchWrap.classList.add("is-open");
    e.searchResults.innerHTML = `<li class="search-empty">No matches for "${escapeHtml(query)}"</li>`;
    return;
  }
  e.searchWrap.classList.add("is-open");
  e.searchResults.innerHTML = hits
    .map(
      (h, i) => `
      <li role="option">
        <button type="button" data-hit="${i}">
          <span class="search-ico">${h.type === "coord" ? "⊹" : "◉"}</span>
          <span class="search-text">
            <span class="search-label">${escapeHtml(h.label)}</span>
            <span class="search-sub">${escapeHtml(h.sublabel || h.type)}</span>
          </span>
        </button>
      </li>`,
    )
    .join("");
  e.searchResults.querySelectorAll<HTMLButtonElement>("button[data-hit]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = Number(btn.dataset.hit);
      flyToHit(hits[idx]);
    });
  });
}

function flyToHit(hit: GeocodeHit): void {
  if (hit.bbox) {
    globe.fitBounds(hit.bbox, { padding: 120, maxZoom: 11 });
  } else {
    globe.flyToBBox(
      {
        west: hit.lng - 0.05,
        east: hit.lng + 0.05,
        south: hit.lat - 0.05,
        north: hit.lat + 0.05,
      },
      { zoom: 10 },
    );
  }
  const e = els();
  if (e.searchInput) e.searchInput.value = hit.label;
  e.searchWrap?.classList.remove("is-open");
  setStatus(`Flew to ${hit.label}. Shift-drag or hit Draw region to define an AOI.`);
}

function escapeHtml(str: string): string {
  return str.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

function setStatus(message: string): void {
  state.status = message;
  updateView();
}

function syncSliderFill(input: HTMLInputElement): void {
  const min = Number(input.min);
  const max = Number(input.max);
  const val = Number(input.value);
  const pct = max > min ? ((val - min) / (max - min)) * 100 : 0;
  input.style.setProperty("--v", `${pct}%`);
}

function updateView(): void {
  const e = els();
  if (!e.status || !e.drawBtn || !e.drawLabel || !e.positiveList || !e.resultList) return;

  e.status.textContent = state.status;
  e.statusPill?.classList.toggle("is-busy", state.loading);

  const armed = globe?.isArmed() ?? false;
  e.drawLabel.textContent = armed ? "Drawing…" : "Draw region";
  e.drawBtn.classList.toggle("is-armed", armed);

  if (e.mShards) e.mShards.textContent = state.manifestShards.length ? String(state.manifestShards.length) : "—";
  if (e.mPatches) e.mPatches.textContent = state.candidateRows.length ? new Intl.NumberFormat().format(state.candidateRows.length) : "—";
  if (e.mRoi) e.mRoi.textContent = state.bbox ? `${(state.bbox.east - state.bbox.west).toFixed(2)}°×${(state.bbox.north - state.bbox.south).toFixed(2)}°` : "—";

  if (e.topkSlider) { e.topkSlider.value = String(state.topK); syncSliderFill(e.topkSlider); }
  if (e.topkValue) e.topkValue.textContent = String(state.topK);

  // View mode tabs
  e.viewTabs.forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.view === state.viewMode);
  });
  if (e.topkControl) e.topkControl.hidden = state.viewMode !== "topk";
  if (e.heatmapLegend) e.heatmapLegend.hidden = state.viewMode !== "heatmap" && state.viewMode !== "contour";
  if (e.outlierLegend) e.outlierLegend.hidden = state.viewMode !== "outlier";
  if (e.thresholdControl) e.thresholdControl.hidden = state.viewMode !== "threshold";

  // Threshold UI
  if (state.viewMode === "threshold" && state.results.length) {
    const scores = state.results.map((r) => r.score);
    const mn = Math.min(...scores);
    const mx = Math.max(...scores);
    if (e.thresholdSlider) {
      e.thresholdSlider.min = String(mn);
      e.thresholdSlider.max = String(mx);
      if (state.threshold === Infinity) state.threshold = scores[Math.min(state.topK, scores.length) - 1] ?? mx;
      e.thresholdSlider.value = String(state.threshold);
      e.thresholdSlider.step = String(Math.max(0.1, (mx - mn) / 200));
      syncSliderFill(e.thresholdSlider);
    }
    if (e.thresholdValue) e.thresholdValue.textContent = state.threshold.toFixed(1);
    const below = state.results.filter((r) => r.score <= state.threshold).length;
    if (e.thresholdCount) e.thresholdCount.textContent = String(below);
    if (e.histogramWrap) e.histogramWrap.innerHTML = renderHistogram(scores, state.threshold);
  }

  if (e.exemplarCount) e.exemplarCount.textContent = String(state.positivePoints.length);
  if (e.clearPointsBtn) e.clearPointsBtn.hidden = state.positivePoints.length === 0;
  if (e.exportBtn) e.exportBtn.hidden = state.results.length === 0;

  // Exemplar list
  e.positiveList.innerHTML = "";
  if (!state.positivePoints.length) {
    e.positiveList.innerHTML = `<li class="empty">No exemplars yet — click inside the AOI to seed the search.</li>`;
  } else {
    for (const [i, p] of state.positivePoints.entries()) {
      const li = document.createElement("li");
      li.className = "exemplar-item";
      li.style.setProperty("--i", String(i));
      li.innerHTML = `
        <button type="button" data-pid="${p.id}">
          <span class="ex-index">E${String(p.id).padStart(2, "0")}</span>
          <span class="ex-coord">${formatLatLng(p.lat, p.lng)}</span>
          <span class="ex-remove" aria-hidden="true">×</span>
        </button>
      `;
      e.positiveList.appendChild(li);
    }
    e.positiveList.querySelectorAll<HTMLButtonElement>("button[data-pid]").forEach((b) => {
      b.addEventListener("click", () => {
        const pid = Number(b.dataset.pid);
        state.positivePoints = state.positivePoints
          .filter((p) => p.id !== pid)
          .map((p, idx) => ({ ...p, id: idx + 1 }));
        globe.setPositives(state.positivePoints);
        void scoreCandidates();
        updateView();
      });
    });
  }

  // Results list — pick the right source and slice for the active view
  const activeResults = state.viewMode === "outlier" ? state.outlierResults : state.results;
  const visible = state.viewMode === "topk"
    ? activeResults.slice(0, state.topK)
    : state.viewMode === "threshold"
      ? activeResults.filter((r) => r.score <= state.threshold)
      : activeResults.slice(0, state.topK); // list shows topK even in heatmap/contour/outlier

  if (e.resultCount) {
    const needsExemplars = state.viewMode !== "outlier";
    if (!state.candidateRows.length) e.resultCount.textContent = "";
    else if (needsExemplars && !state.positivePoints.length) e.resultCount.textContent = "";
    else if (state.viewMode === "threshold")
      e.resultCount.textContent = `${visible.length} / ${new Intl.NumberFormat().format(activeResults.length)} within cutoff`;
    else if (state.viewMode === "topk")
      e.resultCount.textContent = `${visible.length} shown / ${new Intl.NumberFormat().format(activeResults.length)}`;
    else
      e.resultCount.textContent = `${new Intl.NumberFormat().format(activeResults.length)} scored`;
  }

  e.resultList.innerHTML = "";
  if (visible.length) {
    for (const [i, r] of visible.entries()) {
      const c = centroid(r.bbox);
      const li = document.createElement("li");
      li.className = "result-item";
      li.style.setProperty("--i", String(i));
      li.innerHTML = `
        <button type="button" data-chip="${r.chips_id}">
          <span class="rank">${String(i + 1).padStart(2, "0")}</span>
          <span class="rank-body">
            <span class="rank-coord">${formatLatLng(c.lat, c.lng)}</span>
            <span class="rank-chip">${r.chips_id}</span>
          </span>
          <span class="rank-score">${r.score.toFixed(1)}</span>
        </button>
      `;
      e.resultList.appendChild(li);
    }
    e.resultList.querySelectorAll<HTMLButtonElement>("button[data-chip]").forEach((b) => {
      const row = activeResults.find((r) => r.chips_id === b.dataset.chip);
      if (!row) return;
      b.addEventListener("mouseenter", () => globe.setPreview(row));
      b.addEventListener("focus", () => globe.setPreview(row));
      b.addEventListener("mouseleave", () => globe.setPreview(null));
      b.addEventListener("blur", () => globe.setPreview(null));
      b.addEventListener("click", () => globe.flyToBBox(row.bbox, { zoom: 11 }));
    });
  }
}

/* -------------------------------------------------------------- DuckDB layer */

function sqlString(value: string): string {
  return `'${value.replaceAll("'", "''")}'`;
}

function resolveShardUrl(relativePath: string): string {
  try {
    return new URL(relativePath, MANIFEST_URL).toString();
  } catch {
    return relativePath;
  }
}

function buildManifestQuery(bbox: BBox): string {
  return `
    SELECT path, rows, xmin, ymin, xmax, ymax, year
    FROM read_parquet(${sqlString(MANIFEST_URL)})
    WHERE xmax >= ${bbox.west} AND xmin <= ${bbox.east}
      AND ymax >= ${bbox.south} AND ymin <= ${bbox.north}
    ORDER BY rows DESC, path ASC
  `.trim();
}

function buildShardQuery(shardUrl: string, bbox: BBox): string {
  return `
    SELECT chips_id, bbox, embedding
    FROM read_parquet(${sqlString(shardUrl)})
    WHERE bbox.xmax >= ${bbox.west} AND bbox.xmin <= ${bbox.east}
      AND bbox.ymax >= ${bbox.south} AND bbox.ymin <= ${bbox.north}
  `.trim();
}

async function fetchShardCandidates(
  db: duckdb.AsyncDuckDB,
  shardUrl: string,
  bbox: BBox,
): Promise<CandidateRow[]> {
  const conn = await db.connect();
  try {
    const result = await conn.query(buildShardQuery(shardUrl, bbox));
    const rows = result.toArray() as Array<{
      chips_id: string;
      bbox: { xmin: number; ymin: number; xmax: number; ymax: number };
      embedding: unknown;
    }>;
    return rows.map((row) => ({
      chips_id: row.chips_id,
      bbox: normalizeBBox(row.bbox),
      embedding: normalizeEmbedding(row.embedding),
      shard_path: shardUrl,
    }));
  } finally {
    await conn.close();
  }
}

async function mapWithConcurrency<T, R>(
  items: T[],
  limit: number,
  worker: (item: T, index: number) => Promise<R>,
): Promise<PromiseSettledResult<R>[]> {
  const results: PromiseSettledResult<R>[] = new Array(items.length);
  let cursor = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const i = cursor++;
      if (i >= items.length) return;
      try {
        results[i] = { status: "fulfilled", value: await worker(items[i], i) };
      } catch (err) {
        results[i] = { status: "rejected", reason: err };
      }
    }
  });
  await Promise.all(runners);
  return results;
}

async function instantiateDuckDB(): Promise<duckdb.AsyncDuckDB> {
  const bundle = await duckdb.selectBundle(DUCKDB_BUNDLES);
  if (!bundle.mainWorker) throw new Error("DuckDB bundle missing worker");
  const worker = new Worker(bundle.mainWorker);
  const db = new duckdb.AsyncDuckDB(new duckdb.ConsoleLogger(), worker);
  try {
    await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
    const conn = await db.connect();
    await conn.query("INSTALL httpfs; LOAD httpfs;");
    await conn.close();
    return db;
  } catch (err) {
    worker.terminate();
    throw err;
  }
}

function getDuckDB(): Promise<duckdb.AsyncDuckDB> {
  if (!dbPromise) dbPromise = instantiateDuckDB();
  return dbPromise;
}

/* ---------------------------------------------------------- Scoring (worker) */

type WorkerScoreResult = { index: number; score: number };

function ensureScoringWorker(): Worker {
  if (scoringWorker) return scoringWorker;
  scoringWorker = new Worker(new URL("./scoring-worker.ts", import.meta.url), { type: "module" });
  scoringWorker.addEventListener("error", () => {
    scoringWorkerReady = false;
  });
  return scoringWorker;
}

function initScoringWorker(candidates: CandidateRow[]): void {
  const worker = ensureScoringWorker();
  worker.postMessage({
    type: "init",
    embeddings: candidates.map((c) => new Uint8Array(c.embedding)),
  });
  scoringWorkerReady = true;
}

async function scoreWithWorker(exemplars: CandidateRow[]): Promise<RankedRow[]> {
  if (!scoringWorkerReady) return [];
  const worker = ensureScoringWorker();
  const requestId = ++scoringRequestId;
  const excludeIndices = new Set(exemplars.map((ex) => state.candidateRows.indexOf(ex)));
  const results = await new Promise<WorkerScoreResult[]>((resolve, reject) => {
    const onMessage = (event: MessageEvent<{ type: string; requestId: number; results: WorkerScoreResult[] }>) => {
      if (event.data.type !== "score-result" || event.data.requestId !== requestId) return;
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
      resolve(event.data.results);
    };
    const onError = (event: ErrorEvent) => {
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
      scoringWorkerReady = false;
      reject(event.error ?? new Error(event.message));
    };
    worker.addEventListener("message", onMessage);
    worker.addEventListener("error", onError);
    worker.postMessage({
      type: "score",
      requestId,
      exemplars: exemplars.map((ex) => new Uint8Array(ex.embedding)),
      excludeIndices: [...excludeIndices],
    });
  });
  return results.map(({ index, score }) => ({ ...state.candidateRows[index], score }));
}

function resolvePositiveMatches(): PositiveMatch[] {
  return state.positivePoints.flatMap((point) => {
    const intersecting = state.candidateRows.filter((c) => containsPoint(c.bbox, point.lat, point.lng));
    if (!intersecting.length) return [];
    let best = intersecting[0];
    let bestD = Number.POSITIVE_INFINITY;
    for (const option of intersecting) {
      const c = centroid(option.bbox);
      const d = distanceSquared(point.lat, point.lng, c.lat, c.lng);
      if (d < bestD) {
        bestD = d;
        best = option;
      }
    }
    return [{ pointId: point.id, candidate: best }];
  });
}

async function scoreCandidates(): Promise<void> {
  const runId = ++latestScoreRunId;

  // If the region is still downloading, defer scoring. The points are preserved
  // in state.positivePoints and loadRegion() will call scoreCandidates() again
  // once the last shard lands and the scoring worker is ready.
  if (state.loading || !scoringWorkerReady) {
    if (state.positivePoints.length) {
      setStatus(
        `Queued ${state.positivePoints.length} exemplar(s) — waiting for shards to finish downloading…`,
      );
    }
    return;
  }

  if (!state.candidateRows.length || !state.positivePoints.length) {
    state.positiveMatches = [];
    state.results = [];
    globe.setPositiveMatches([]);
    globe.setResults([], state.topK, state.viewMode);
    globe.setPreview(null);
    updateView();
    return;
  }

  state.positiveMatches = resolvePositiveMatches();
  const exemplars = state.positiveMatches.map((m) => m.candidate);
  globe.setPositiveMatches(state.positiveMatches);

  if (!exemplars.length) {
    state.results = [];
    globe.setResults([], state.topK, state.viewMode);
    setStatus("No patch under the selected point — try closer to a tile center.");
    return;
  }

  setStatus(`Scoring ${new Intl.NumberFormat().format(state.candidateRows.length)} candidates…`);
  const scored = await scoreWithWorker(exemplars);
  if (runId !== latestScoreRunId) return;
  state.results = scored;
  globe.setResults(scored, state.topK, state.viewMode);
  setStatus(`Ranked ${scored.length} candidates against ${exemplars.length} exemplar(s).`);
  updateView();
}

/* ---------------------------------------------------------- Histogram */

function renderHistogram(scores: number[], threshold: number): string {
  if (!scores.length) return "";
  const bins = 32;
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const range = max - min || 1;
  const binWidth = range / bins;
  const counts = new Array(bins).fill(0) as number[];
  for (const s of scores) {
    const bin = Math.min(Math.floor((s - min) / binWidth), bins - 1);
    counts[bin]++;
  }
  const maxCount = Math.max(...counts);
  const w = 240;
  const h = 36;
  const bw = w / bins;
  const bars = counts
    .map((count, i) => {
      const binMid = min + (i + 0.5) * binWidth;
      const barH = maxCount > 0 ? (count / maxCount) * h : 0;
      const x = i * bw;
      const y = h - barH;
      const fill = binMid <= threshold ? "var(--accent)" : "rgba(243,236,216,0.1)";
      return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(bw - 0.5).toFixed(1)}" height="${barH.toFixed(1)}" fill="${fill}" rx="1"/>`;
    })
    .join("");
  return `<svg class="histogram-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">${bars}</svg>`;
}

/* ---------------------------------------------------------- Outlier scoring */

async function computeOutliers(): Promise<void> {
  if (!scoringWorkerReady || !state.candidateRows.length) return;
  if (state.outlierComputed) return;

  setStatus("Computing outlier scores…");
  const worker = ensureScoringWorker();
  const requestId = ++scoringRequestId;
  const results = await new Promise<WorkerScoreResult[]>((resolve, reject) => {
    const onMessage = (event: MessageEvent<{ type: string; requestId: number; results: WorkerScoreResult[] }>) => {
      if (event.data.type !== "outlier-result" || event.data.requestId !== requestId) return;
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
      resolve(event.data.results);
    };
    const onError = (event: ErrorEvent) => {
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
      reject(event.error ?? new Error(event.message));
    };
    worker.addEventListener("message", onMessage);
    worker.addEventListener("error", onError);
    worker.postMessage({ type: "outlier", requestId, sampleSize: 200 });
  });

  state.outlierResults = results.map(({ index, score }) => ({ ...state.candidateRows[index], score }));
  state.outlierComputed = true;
  globe.setResults(state.outlierResults, state.topK, "outlier");
  setStatus(`Outlier analysis: ${state.outlierResults.length} patches scored. Brightest = most unique.`);
  updateView();
}

/* ---------------------------------------------------------- GeoParquet export */

async function exportGeoParquet(): Promise<void> {
  const exemplars = state.positiveMatches.map((m) => m.candidate);
  const topk = state.results.slice(0, state.viewMode === "topk" ? state.topK : state.results.length);
  if (!exemplars.length && !topk.length) return;

  setStatus("Exporting GeoParquet…");
  try {
    const db = await getDuckDB();
    const conn = await db.connect();

    // Build VALUES clauses for exemplars and candidates
    const rows: string[] = [];
    for (const ex of exemplars) {
      const c = centroid(ex.bbox);
      rows.push(
        `('exemplar', ${sqlString(ex.chips_id)}, ${c.lat}, ${c.lng}, ${ex.bbox.west}, ${ex.bbox.south}, ${ex.bbox.east}, ${ex.bbox.north}, NULL::DOUBLE, NULL::INT)`,
      );
    }
    for (const [i, r] of topk.entries()) {
      const c = centroid(r.bbox);
      rows.push(
        `('candidate', ${sqlString(r.chips_id)}, ${c.lat}, ${c.lng}, ${r.bbox.west}, ${r.bbox.south}, ${r.bbox.east}, ${r.bbox.north}, ${r.score}, ${i + 1})`,
      );
    }

    const vfsPath = "/tmp/export.parquet";
    await conn.query(`
      COPY (
        SELECT * FROM (
          VALUES ${rows.join(",\n")}
        ) AS t(type, chips_id, lat, lng, west, south, east, north, score, rank)
      ) TO '${vfsPath}' (FORMAT PARQUET, COMPRESSION ZSTD)
    `);

    const buf = await db.copyFileToBuffer(vfsPath);
    await conn.close();

    // Trigger download
    const blob = new Blob([buf.buffer as ArrayBuffer], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `terrabit-export-${Date.now()}.parquet`;
    a.click();
    URL.revokeObjectURL(url);
    setStatus(`Exported ${exemplars.length} exemplar(s) + ${topk.length} candidates to GeoParquet.`);
  } catch (err) {
    setStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

/* ---------------------------------------------------------- App actions */

async function loadRegion(bbox: BBox): Promise<void> {
  const runId = ++latestLoadRunId;
  // Clear old state IMMEDIATELY so the user never sees stale data when redrawing.
  state.bbox = bbox;
  state.positivePoints = [];
  state.positiveMatches = [];
  state.results = [];
  state.outlierResults = [];
  state.outlierComputed = false;
  state.threshold = Infinity;
  state.candidateRows = [];
  state.manifestShards = [];
  state.loading = true;
  globe.setAoi(bbox);
  globe.setPositives([]);
  globe.setPositiveMatches([]);
  globe.setResults([], state.topK, state.viewMode);
  globe.setPreview(null);
  setStatus("Fetching intersecting shards…");
  updateView();

  try {
    const db = await getDuckDB();
    const conn = await db.connect();
    const manifestResult = await conn.query(buildManifestQuery(bbox));
    const shards = manifestResult.toArray() as ManifestRow[];
    await conn.close();
    if (runId !== latestLoadRunId) return;

    state.manifestShards = shards;
    updateView();
    if (!shards.length) {
      state.loading = false;
      setStatus("No shards intersect that region.");
      return;
    }

    setStatus(`Loading patches from ${shards.length} shard(s)…`);
    globe.fitBounds(bbox, { padding: 60 });

    // Query each shard in parallel with bounded concurrency. This is much more
    // reliable than a single read_parquet([url1, url2, ...]) call — one slow or
    // flaky shard used to stall / truncate the whole batch.
    let completed = 0;
    const shardUrls = shards.map((s) => resolveShardUrl(s.path));
    const all: CandidateRow[] = [];
    const settled = await mapWithConcurrency(shardUrls, 8, async (url) => {
      const rows = await fetchShardCandidates(db, url, bbox);
      if (runId !== latestLoadRunId) return rows;
      all.push(...rows);
      completed += 1;
      // Live progress update
      state.candidateRows = all;
      setStatus(`Loading patches — ${completed}/${shards.length} shards · ${new Intl.NumberFormat().format(all.length)} patches`);
      return rows;
    });
    if (runId !== latestLoadRunId) return;

    const failed = settled.filter((r) => r.status === "rejected");
    if (failed.length) {
      console.warn("Shard fetch failures:", failed);
    }

    state.candidateRows = all;
    initScoringWorker(state.candidateRows);
    state.loading = false;
    const base = state.candidateRows.length
      ? `Region loaded — ${new Intl.NumberFormat().format(state.candidateRows.length)} patches from ${shards.length - failed.length}/${shards.length} shards. Click inside the AOI to seed an exemplar.`
      : "Region loaded, but no patches returned.";
    setStatus(failed.length ? `${base} (${failed.length} shard(s) failed)` : base);
    updateView();

    // If the user clicked exemplar points while shards were still streaming,
    // they were queued — run the scoring pass now that everything is ready.
    if (state.positivePoints.length && state.candidateRows.length) {
      void scoreCandidates();
    }
  } catch (err) {
    if (runId !== latestLoadRunId) return;
    state.loading = false;
    setStatus(`Failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

function pickZoomForBBox(bbox: BBox): number {
  const span = Math.max(bbox.east - bbox.west, bbox.north - bbox.south);
  if (span > 20) return 3;
  if (span > 8) return 4.5;
  if (span > 3) return 6;
  if (span > 1) return 7.5;
  if (span > 0.3) return 9;
  return 10.5;
}

function findNearestCandidate(lat: number, lng: number): CandidateRow | null {
  const intersecting = state.candidateRows.filter((c) => containsPoint(c.bbox, lat, lng));
  if (!intersecting.length) return null;
  let best = intersecting[0];
  let bestD = Number.POSITIVE_INFINITY;
  for (const option of intersecting) {
    const c = centroid(option.bbox);
    const d = distanceSquared(lat, lng, c.lat, c.lng);
    if (d < bestD) { bestD = d; best = option; }
  }
  return best;
}

function addPositive(lat: number, lng: number): void {
  // Deduplicate: skip if this click resolves to an already-selected patch
  const patch = findNearestCandidate(lat, lng);
  if (patch) {
    const existing = state.positivePoints.some((p) => {
      const ep = findNearestCandidate(p.lat, p.lng);
      return ep && ep.chips_id === patch.chips_id;
    });
    if (existing) return;
  }
  state.positivePoints.push({ id: state.positivePoints.length + 1, lat, lng });
  globe.setPositives(state.positivePoints);
  void scoreCandidates();
  updateView();
}

function clearPoints(): void {
  if (!state.positivePoints.length) return;
  state.positivePoints = [];
  state.positiveMatches = [];
  state.results = [];
  globe.setPositives([]);
  globe.setPositiveMatches([]);
  globe.setResults([], state.topK, state.viewMode);
  globe.setPreview(null);
  setStatus("Exemplar points cleared. Click inside the AOI to seed new ones.");
  updateView();
}

function clearRegion(): void {
  state.bbox = null;
  state.manifestShards = [];
  state.candidateRows = [];
  state.positivePoints = [];
  state.positiveMatches = [];
  state.results = [];
  state.topK = DEFAULT_TOP_K;
  state.viewMode = "topk";
  state.outlierResults = [];
  state.outlierComputed = false;
  state.threshold = Infinity;
  globe.setAoi(null);
  globe.setPositives([]);
  globe.setPositiveMatches([]);
  globe.setResults([], state.topK, state.viewMode);
  globe.setPreview(null);
  setStatus("Cleared. Shift-drag to define a new region.");
}

/* --------------------------------------------------------------- Bootstrap */

function wire(): void {
  const e = els();
  e.drawBtn?.addEventListener("click", () => {
    globe.armDraw(!globe.isArmed());
    setStatus(globe.isArmed() ? "Draw armed — drag on the globe to define a region." : "Draw disarmed.");
  });
  e.clearRegionBtn?.addEventListener("click", clearRegion);
  e.clearPointsBtn?.addEventListener("click", clearPoints);
  e.clearPointsBtn2?.addEventListener("click", clearPoints);
  e.exportBtn?.addEventListener("click", () => void exportGeoParquet());
  e.topkSlider?.addEventListener("input", (ev) => {
    state.topK = Number((ev.currentTarget as HTMLInputElement).value);
    globe.setResults(state.results, state.topK, state.viewMode);
    updateView();
  });
  e.thresholdSlider?.addEventListener("input", (ev) => {
    state.threshold = Number((ev.currentTarget as HTMLInputElement).value);
    const filtered = state.results.filter((r) => r.score <= state.threshold);
    globe.setResults(filtered, filtered.length, "threshold");
    updateView();
  });

  // View mode tabs
  e.viewTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const mode = tab.dataset.view as ViewMode;
      state.viewMode = mode;
      if (mode === "outlier" && !state.outlierComputed) {
        void computeOutliers();
      } else if (mode === "outlier") {
        globe.setResults(state.outlierResults, state.topK, mode);
      } else if (mode === "threshold") {
        const filtered = state.results.filter((r) => r.score <= state.threshold);
        globe.setResults(filtered, filtered.length, mode);
      } else {
        globe.setResults(state.results, state.topK, mode);
      }
      updateView();
    });
  });

  // Geocoder search
  e.searchInput?.addEventListener("input", (ev) => {
    const q = (ev.currentTarget as HTMLInputElement).value.trim();
    if (geocodeTimer) window.clearTimeout(geocodeTimer);
    if (!q) {
      renderGeocodeResults([], "");
      e.searchSpinner?.classList.remove("is-busy");
      return;
    }
    // Instant coord parse
    const coord = tryParseLatLng(q);
    const reqId = ++geocodeReqId;
    e.searchSpinner?.classList.add("is-busy");
    geocodeTimer = window.setTimeout(async () => {
      try {
        const hits = await fetchGeocode(q);
        if (reqId !== geocodeReqId) return;
        const combined = coord ? [coord, ...hits] : hits;
        renderGeocodeResults(combined, q);
      } catch {
        if (reqId !== geocodeReqId) return;
        renderGeocodeResults(coord ? [coord] : [], q);
      } finally {
        if (reqId === geocodeReqId) e.searchSpinner?.classList.remove("is-busy");
      }
    }, 320);
  });

  e.searchInput?.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      const first = e.searchResults?.querySelector<HTMLButtonElement>("button[data-hit]");
      first?.click();
    }
    if (ev.key === "Escape") {
      (ev.currentTarget as HTMLInputElement).blur();
      e.searchWrap?.classList.remove("is-open");
    }
  });

  e.searchInput?.addEventListener("focus", () => {
    if (e.searchInput && e.searchInput.value) e.searchWrap?.classList.add("is-open");
  });

  // Click outside to close dropdown
  document.addEventListener("click", (ev) => {
    const t = ev.target as Node;
    if (e.searchWrap && !e.searchWrap.contains(t)) e.searchWrap.classList.remove("is-open");
  });

  // Global "/" shortcut to focus search
  window.addEventListener("keydown", (ev) => {
    if (ev.key === "/" && document.activeElement !== e.searchInput) {
      ev.preventDefault();
      e.searchInput?.focus();
      return;
    }
    if (ev.key !== "Escape") return;
    if (globe.isArmed()) {
      globe.armDraw(false);
      globe.cancelDraft();
      setStatus("Draw disarmed.");
      return;
    }
    if (state.bbox) clearRegion();
  });
}

function bootstrap(): void {
  renderShell();
  const mapEl = document.querySelector<HTMLDivElement>("#map");
  if (!mapEl) throw new Error("#map missing");
  globe = new GlobeMap(mapEl, {
    onDrawComplete: (bbox) => {
      void loadRegion(bbox);
    },
    onAoiClick: (lat, lng) => addPositive(lat, lng),
    onResultHover: (row) => globe.setPreview(row),
    onResultPick: (row) => {
      const c = centroid(row.bbox);
      addPositive(c.lat, c.lng);
    },
    getBBox: () => state.bbox,
    getResults: () => state.results,
    getTopK: () => state.topK,
  });
  wire();
  renderAoiPresets();
  updateView();
}

bootstrap();
