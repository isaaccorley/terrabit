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
  CombineMethod,
  ManifestRow,
  NegativePoint,
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
  "https://data.source.coop/geospatialml/terrabit/clay-v1_5-binary-sentinel-2/manifest.parquet";
const DEFAULT_TOP_K = 25;
const MAX_TOP_K = 100;

type AppState = {
  bbox: BBox | null;
  status: string;
  manifestShards: ManifestRow[];
  candidateRows: CandidateRow[];
  positivePoints: PositivePoint[];
  negativePoints: NegativePoint[];
  positiveMatches: PositiveMatch[];
  results: RankedRow[];
  outlierResults: RankedRow[];
  outlierComputed: boolean;
  surpriseResults: RankedRow[];
  surpriseComputed: boolean;
  gradientResults: RankedRow[];
  topK: number;
  viewMode: ViewMode;
  threshold: number;
  overlayVisible: boolean;
  loading: boolean;
  combineMethod: CombineMethod;
  invertSearch: boolean;
};

const state: AppState = {
  bbox: null,
  status: "Spin the globe. Zoom. Shift-drag or hit Draw region to define an AOI.",
  manifestShards: [],
  candidateRows: [],
  positivePoints: [],
  negativePoints: [],
  positiveMatches: [],
  results: [],
  outlierResults: [],
  outlierComputed: false,
  surpriseResults: [],
  surpriseComputed: false,
  gradientResults: [],
  topK: DEFAULT_TOP_K,
  viewMode: "topk",
  threshold: Infinity,
  overlayVisible: true,
  loading: false,
  combineMethod: "mean",
  invertSearch: false,
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
                <button id="invert-toggle" class="icon-btn" type="button" title="Invert search (find opposites)" aria-label="Invert search">
                  <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="8" cy="8" r="6"/><line x1="3" y1="3" x2="13" y2="13"/></svg>
                </button>
                <select id="combine-method" class="combine-select" title="Combine method">
                  <option value="mean">Mean</option>
                  <option value="and">AND</option>
                  <option value="or">OR</option>
                  <option value="xor">XOR</option>
                </select>
                <span id="exemplar-count" class="count-badge">0</span>
                <button id="clear-points-btn" class="icon-btn" type="button" title="Clear points" aria-label="Clear points">
                  <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>
                </button>
              </div>
            </header>
            <ol id="positive-list" class="exemplar-list"></ol>
            <div id="negative-section" hidden>
              <div class="neg-header">
                <span class="panel-kicker neg-kicker">Negatives</span>
                <button id="clear-negatives-btn" class="icon-btn" type="button" title="Clear negatives" aria-label="Clear negatives">
                  <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>
                </button>
              </div>
              <ol id="negative-list" class="exemplar-list negative-list"></ol>
            </div>
          </div>

          <div class="sub-card sub-card-retrieval">
            <header class="sub-head sub-head-stack">
              <div>
                <span class="panel-kicker">03 · Retrieval</span>

              </div>
              <div class="sub-head-actions">
                <span id="result-count" class="result-summary"></span>
                <button class="icon-btn" type="button" aria-label="View mode help" data-help="retrieval">
                  <svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="8" cy="8" r="6.5"/><path d="M6.2 6.1a2 2 0 0 1 3.6 1.2c0 1.5-1.8 1.8-1.8 3"/><circle cx="8" cy="12" r=".7" fill="currentColor" stroke="none"/></svg>
                </button>
                <button id="overlay-toggle" class="icon-btn is-on" type="button" title="Toggle map overlay" aria-label="Toggle map overlay">
                  <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M1 8l7-5 7 5-7 5z"/><path d="M1 11l7 5 7-5" opacity=".4"/></svg>
                </button>
              </div>
            </header>

            <div class="view-toggle" role="tablist" aria-label="Result view">
              <button data-view="topk" class="view-tab is-active" type="button" role="tab" title="Top-K — Ranked list of most similar patches" aria-label="Top-K ranked list">
                <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="2" y="3" width="12" height="3" rx="0.6"/><rect x="2" y="7" width="9" height="3" rx="0.6"/><rect x="2" y="11" width="5" height="3" rx="0.6"/></svg>
                <span class="view-label">Top-K</span>
              </button>
              <button data-view="heatmap" class="view-tab" type="button" role="tab" title="Heatmap — Continuous similarity surface across all patches" aria-label="Heatmap of all scored tiles">
                <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="2" y="2" width="4" height="4"/><rect x="6" y="2" width="4" height="4"/><rect x="10" y="2" width="4" height="4"/><rect x="2" y="6" width="4" height="4"/><rect x="6" y="6" width="4" height="4"/><rect x="10" y="6" width="4" height="4"/><rect x="2" y="10" width="4" height="4"/><rect x="6" y="10" width="4" height="4"/><rect x="10" y="10" width="4" height="4"/></svg>
                <span class="view-label">Heat</span>
              </button>
              <button data-view="threshold" class="view-tab" type="button" role="tab" title="Cutoff — Show all patches within a Hamming distance threshold" aria-label="Distance cutoff">
                <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="2" y="10" width="2" height="4"/><rect x="5" y="6" width="2" height="8"/><rect x="8" y="3" width="2" height="11"/><rect x="11" y="8" width="2" height="6"/><line x1="1" y1="7" x2="15" y2="7" stroke-dasharray="2 1.5"/></svg>
                <span class="view-label">Cutoff</span>
              </button>
              <button data-view="outlier" class="view-tab" type="button" role="tab" title="Outlier — Find the most unique and unusual patches in the region" aria-label="Most unique patches">
                <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="8" cy="8" r="2"/><circle cx="4" cy="6" r="1.2"/><circle cx="12" cy="5" r="1.2"/><circle cx="13" cy="11" r="1.2"/><circle cx="3" cy="12" r="1.2"/></svg>
                <span class="view-label">Outlier</span>
              </button>
              <button data-view="surprise" class="view-tab" type="button" role="tab" title="Surprise — Patches that look different from their geographic neighbors" aria-label="Spatial surprise">
                <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M8 2v7"/><circle cx="8" cy="12.5" r="1.5"/></svg>
                <span class="view-label">Surprise</span>
              </button>
              <button data-view="gradient" class="view-tab" type="button" role="tab" title="Edge — Detect boundaries where similarity scores change sharply" aria-label="Similarity gradient edge detection">
                <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M2 14L8 2l6 12"/><line x1="4" y1="10" x2="12" y2="10"/></svg>
                <span class="view-label">Edge</span>
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
                <span>Distant</span>
                <span>Similar</span>
              </div>
            </div>

            <div id="outlier-legend" class="heatmap-legend" hidden>
              <div class="legend-gradient legend-gradient-outlier"></div>
              <div class="legend-labels">
                <span>Common</span>
                <span>Unique</span>
              </div>
            </div>

            <div id="surprise-legend" class="heatmap-legend" hidden>
              <div class="legend-gradient legend-gradient-outlier"></div>
              <div class="legend-labels">
                <span>Expected</span>
                <span>Surprising</span>
              </div>
            </div>

            <div id="gradient-legend" class="heatmap-legend" hidden>
              <div class="legend-gradient legend-gradient-outlier"></div>
              <div class="legend-labels">
                <span>Uniform</span>
                <span>Boundary</span>
              </div>
            </div>

            <div id="gradient-msg" class="gradient-msg" hidden>
              <span class="hint">Add exemplars to use Edge view.</span>
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
            <button id="fingerprint-btn" class="btn btn-sm btn-ghost btn-export" type="button" hidden>
              <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="8" cy="8" r="6"/><path d="M8 2a6 6 0 0 1 0 12"/><path d="M8 5a3 3 0 0 1 0 6"/></svg>
              <span>Find similar regions</span>
            </button>
            <div id="fingerprint-results" class="fingerprint-results" hidden></div>
          </div>
        </div>
      </section>

      <nav class="hud-panel hud-panel-right hud-aoi-nav" id="aoi-nav">
        <header class="panel-head panel-head-row">
          <span class="panel-kicker">Explore</span>
          <button class="icon-btn" type="button" aria-label="About Explore" data-help="explore">
            <svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="8" cy="8" r="6.5"/><path d="M6.2 6.1a2 2 0 0 1 3.6 1.2c0 1.5-1.8 1.8-1.8 3"/><circle cx="8" cy="12" r=".7" fill="currentColor" stroke="none"/></svg>
          </button>
        </header>
        <ul id="aoi-list" class="aoi-list"></ul>
        <div class="ip-divider"></div>
        <section class="ip-section">
          <header class="ip-head">
            <span class="panel-kicker">Discoveries</span>
            <button class="icon-btn" type="button" aria-label="About Discoveries" data-help="discoveries">
              <svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="8" cy="8" r="6.5"/><path d="M6.2 6.1a2 2 0 0 1 3.6 1.2c0 1.5-1.8 1.8-1.8 3"/><circle cx="8" cy="12" r=".7" fill="currentColor" stroke="none"/></svg>
            </button>
          </header>
          <ul id="ip-list" class="ip-list"></ul>
        </section>
      </nav>

    </div>
    <button id="tutorial-trigger" class="tutorial-trigger" type="button" aria-label="Open tutorial" title="Help &amp; tutorial">?</button>
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
    negativeList: document.querySelector<HTMLOListElement>("#negative-list"),
    negativeSection: document.querySelector<HTMLElement>("#negative-section"),
    clearNegativesBtn: document.querySelector<HTMLButtonElement>("#clear-negatives-btn"),
    exemplarCount: document.querySelector<HTMLElement>("#exemplar-count"),
    invertToggle: document.querySelector<HTMLButtonElement>("#invert-toggle"),
    combineSelect: document.querySelector<HTMLSelectElement>("#combine-method"),
    topkSlider: document.querySelector<HTMLInputElement>("#topk-slider"),
    topkValue: document.querySelector<HTMLElement>("#topk-value"),
    topkControl: document.querySelector<HTMLElement>("#topk-control"),
    viewTabs: document.querySelectorAll<HTMLButtonElement>(".view-tab[data-view]"),
    heatmapLegend: document.querySelector<HTMLElement>("#heatmap-legend"),
    outlierLegend: document.querySelector<HTMLElement>("#outlier-legend"),
    surpriseLegend: document.querySelector<HTMLElement>("#surprise-legend"),
    gradientLegend: document.querySelector<HTMLElement>("#gradient-legend"),
    gradientMsg: document.querySelector<HTMLElement>("#gradient-msg"),
    thresholdControl: document.querySelector<HTMLElement>("#threshold-control"),
    thresholdSlider: document.querySelector<HTMLInputElement>("#threshold-slider"),
    thresholdValue: document.querySelector<HTMLElement>("#threshold-value"),
    thresholdCount: document.querySelector<HTMLElement>("#threshold-count"),
    histogramWrap: document.querySelector<HTMLElement>("#histogram-wrap"),
    overlayToggle: document.querySelector<HTMLButtonElement>("#overlay-toggle"),
    clearPointsBtn: document.querySelector<HTMLButtonElement>("#clear-points-btn"),
    resultCount: document.querySelector<HTMLElement>("#result-count"),
    resultList: document.querySelector<HTMLOListElement>("#result-list"),
    exportBtn: document.querySelector<HTMLButtonElement>("#export-btn"),
    fingerprintBtn: document.querySelector<HTMLButtonElement>("#fingerprint-btn"),
    fingerprintResults: document.querySelector<HTMLElement>("#fingerprint-results"),
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
  { name: "Malé Atoll", tag: "coral atoll", bbox: { west: 73.35, south: 4.05, east: 73.7, north: 4.35 } },
  { name: "Rotterdam", tag: "port", bbox: { west: 3.85, south: 51.85, east: 4.35, north: 52.05 } },
  { name: "Atacama", tag: "lithium mines", bbox: { west: -68.6, south: -23.8, east: -67.8, north: -23.1 } },
  { name: "Center pivots", tag: "agriculture", bbox: { west: 37.2, south: 29.0, east: 39.0, north: 30.2 } },
  { name: "Palm Islands", tag: "coastal eng.", bbox: { west: 54.95, south: 25.05, east: 55.25, north: 25.2 } },
  { name: "Bhadla Solar", tag: "solar farm", bbox: { west: 71.6, south: 27.3, east: 72.1, north: 27.65 } },
  { name: "Rondônia", tag: "deforestation", bbox: { west: -63.5, south: -11.0, east: -62.5, north: -10.0 } },
  { name: "Kansas Grid", tag: "cropland", bbox: { west: -101.0, south: 37.6, east: -100.0, north: 38.4 } },
  { name: "Hartsfield ATL", tag: "airport", bbox: { west: -84.55, south: 33.55, east: -84.35, north: 33.7 } },
  { name: "Pilbara Mines", tag: "mining", bbox: { west: 117.6, south: -22.8, east: 118.8, north: -21.8 } },
  { name: "Venice Lagoon", tag: "lagoon", bbox: { west: 12.15, south: 45.3, east: 12.55, north: 45.55 } },
  { name: "Vatnajökull", tag: "glacier", bbox: { west: -17.2, south: 64.1, east: -16.2, north: 64.6 } },
  { name: "Ganges Delta", tag: "river delta", bbox: { west: 89.0, south: 21.6, east: 90.0, north: 22.4 } },
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

type InterestingCategory = "temporal" | "outlier" | "diverse" | "cluster" | "entropy";
type InterestingPoint = { name: string; tag: string; bbox: BBox; category: InterestingCategory };

const INTERESTING_POINTS: InterestingPoint[] = [
  // non-polar temporal change hotspots (2024→2025)
  { name: "Moosonee", tag: "boreal Δ", category: "temporal", bbox: { west: -80.98, south: 59.82, east: -80.38, north: 60.42 } },
  { name: "Comoros", tag: "tropical Δ", category: "temporal", bbox: { west: 43.72, south: -11.21, east: 44.32, north: -10.61 } },
  { name: "SE Tasmania", tag: "island Δ", category: "temporal", bbox: { west: 149.03, south: -40.87, east: 149.63, north: -40.27 } },
  // k-NN isolation outliers — patches with no close relatives in the full dataset
  { name: "Krasnoyarsk", tag: "isolated", category: "outlier", bbox: { west: 92.71, south: 60.35, east: 93.31, north: 60.95 } },
  { name: "W Siberia", tag: "isolated", category: "outlier", bbox: { west: 77.54, south: 56.71, east: 78.14, north: 57.31 } },
  { name: "Dead Sea", tag: "isolated", category: "outlier", bbox: { west: 35.14, south: 30.82, east: 35.74, north: 31.42 } },
  { name: "Thar Desert", tag: "isolated", category: "outlier", bbox: { west: 71.70, south: 26.50, east: 72.30, north: 27.10 } },
  { name: "Iceland Lava", tag: "isolated", category: "outlier", bbox: { west: -17.52, south: 63.52, east: -16.92, north: 64.12 } },
  // rare cluster types (smallest / rarest surface types globally)
  { name: "Mauritanian Erg", tag: "rare surface", category: "cluster", bbox: { west: -7.41, south: 19.79, east: -6.81, north: 20.39 } },
  { name: "St. Elias Mtn", tag: "rare surface", category: "cluster", bbox: { west: -139.26, south: 59.13, east: -138.66, north: 59.73 } },
  { name: "Karakum Desert", tag: "rare surface", category: "cluster", bbox: { west: 61.00, south: 38.41, east: 61.60, north: 39.01 } },
  { name: "Amazon", tag: "rare surface", category: "cluster", bbox: { west: -64.66, south: -5.83, east: -64.06, north: -5.23 } },
  { name: "Lake Turkana", tag: "rare surface", category: "cluster", bbox: { west: 34.82, south: 2.31, east: 35.42, north: 2.91 } },
  { name: "Richat Structure", tag: "rare surface", category: "cluster", bbox: { west: -11.70, south: 20.80, east: -11.10, north: 21.40 } },
  { name: "Salar de Uyuni", tag: "rare surface", category: "cluster", bbox: { west: -68.00, south: -20.60, east: -67.40, north: -20.00 } },
  { name: "Namib Sand Sea", tag: "rare surface", category: "cluster", bbox: { west: 14.70, south: -25.00, east: 15.30, north: -24.40 } },
  { name: "Sundarbans", tag: "rare surface", category: "cluster", bbox: { west: 88.90, south: 21.60, east: 89.50, north: 22.20 } },
  { name: "Tibetan Plateau", tag: "rare surface", category: "cluster", bbox: { west: 85.70, south: 30.20, east: 86.30, north: 30.80 } },
  { name: "Chott el Djerid", tag: "rare surface", category: "cluster", bbox: { west: 8.20, south: 33.50, east: 8.80, north: 34.10 } },
  { name: "Danakil Depression", tag: "rare surface", category: "cluster", bbox: { west: 40.50, south: 13.80, east: 41.10, north: 14.40 } },
  // high bit-entropy (information-rich)
  { name: "Bering Sea", tag: "high entropy", category: "entropy", bbox: { west: -170.79, south: 63.20, east: -170.19, north: 63.80 } },
  { name: "Yamal", tag: "high entropy", category: "entropy", bbox: { west: 63.22, south: 76.19, east: 63.82, north: 76.79 } },
  { name: "Adelaide Hills", tag: "high entropy", category: "entropy", bbox: { west: 138.20, south: -35.79, east: 138.80, north: -35.19 } },
  { name: "Okavango Delta", tag: "high entropy", category: "entropy", bbox: { west: 22.60, south: -19.80, east: 23.20, north: -19.20 } },
  { name: "Inner Niger Delta", tag: "high entropy", category: "entropy", bbox: { west: -4.50, south: 14.70, east: -3.90, north: 15.30 } },
  { name: "Pantanal", tag: "high entropy", category: "entropy", bbox: { west: -57.80, south: -17.80, east: -57.20, north: -17.20 } },
  { name: "Mekong Delta", tag: "high entropy", category: "entropy", bbox: { west: 105.20, south: 10.00, east: 105.80, north: 10.60 } },
  { name: "Irrawaddy Delta", tag: "high entropy", category: "entropy", bbox: { west: 95.00, south: 15.50, east: 95.60, north: 16.10 } },
  { name: "Lena Delta", tag: "high entropy", category: "entropy", bbox: { west: 126.20, south: 72.40, east: 126.80, north: 73.00 } },
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

function renderInterestingPoints(): void {
  const list = document.querySelector<HTMLUListElement>("#ip-list");
  if (!list) return;
  list.innerHTML = INTERESTING_POINTS.map(
    (pt, i) => `
    <li class="ip-item" data-category="${pt.category}" style="--i:${i}">
      <button type="button" data-ip="${i}">
        <span class="ip-name">${pt.name}</span>
        <span class="ip-tag">${pt.tag}</span>
      </button>
    </li>`,
  ).join("");
  list.querySelectorAll<HTMLButtonElement>("button[data-ip]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const pt = INTERESTING_POINTS[Number(btn.dataset.ip)];
      globe.fitBounds(pt.bbox, { padding: 60, maxZoom: 11 });
      setStatus(`${pt.name} — shift-drag to draw a region.`);
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

function compactNum(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
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
  if (e.heatmapLegend) e.heatmapLegend.hidden = state.viewMode !== "heatmap";
  if (e.outlierLegend) e.outlierLegend.hidden = state.viewMode !== "outlier";
  if (e.surpriseLegend) e.surpriseLegend.hidden = state.viewMode !== "surprise";
  if (e.gradientLegend) e.gradientLegend.hidden = state.viewMode !== "gradient" || !state.results.length;
  if (e.gradientMsg) e.gradientMsg.hidden = state.viewMode !== "gradient" || state.results.length > 0;
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
  if (e.fingerprintBtn) e.fingerprintBtn.hidden = !(state.bbox && state.positivePoints.length > 0 && state.candidateRows.length > 0);
  e.overlayToggle?.classList.toggle("is-on", state.overlayVisible);
  e.invertToggle?.classList.toggle("is-on", state.invertSearch);
  if (e.combineSelect) {
    e.combineSelect.value = state.combineMethod;
    e.combineSelect.hidden = state.positivePoints.length < 2;
  }

  // Negative section
  if (e.negativeSection) e.negativeSection.hidden = state.negativePoints.length === 0;
  if (e.negativeList && state.negativePoints.length) {
    e.negativeList.innerHTML = "";
    for (const [i, p] of state.negativePoints.entries()) {
      const li = document.createElement("li");
      li.className = "exemplar-item neg-item";
      li.style.setProperty("--i", String(i));
      li.innerHTML = `
        <button type="button" data-nid="${p.id}">
          <span class="ex-index neg-index">N${String(p.id).padStart(2, "0")}</span>
          <span class="ex-coord">${formatLatLng(p.lat, p.lng)}</span>
          <span class="ex-remove" aria-hidden="true">\u00d7</span>
        </button>
      `;
      e.negativeList.appendChild(li);
    }
    e.negativeList.querySelectorAll<HTMLButtonElement>("button[data-nid]").forEach((b) => {
      b.addEventListener("click", (ev) => {
        const target = ev.target as HTMLElement;
        const nid = Number(b.dataset.nid);
        if (target.closest(".ex-remove")) {
          state.negativePoints = state.negativePoints
            .filter((p) => p.id !== nid)
            .map((p, idx) => ({ ...p, id: idx + 1 }));
          globe.setNegatives(state.negativePoints);
          void scoreCandidates();
          updateView();
        } else {
          const pt = state.negativePoints.find((p) => p.id === nid);
          if (pt) globe.map.flyTo({ center: [pt.lng, pt.lat], zoom: 13 });
        }
      });
    });
  }

  // Exemplar list
  e.positiveList.innerHTML = "";
  if (!state.positivePoints.length) {
    e.positiveList.innerHTML = `<li class="empty">No exemplars yet — click anywhere on the map to seed the search.</li>`;
  } else {
    for (const [i, p] of state.positivePoints.entries()) {
      const li = document.createElement("li");
      li.className = "exemplar-item";
      li.style.setProperty("--i", String(i));
      const isExternal = Boolean(p.embedding);
      li.innerHTML = `
        <button type="button" data-pid="${p.id}">
          <span class="ex-index">${isExternal ? "⊕" : "E"}${String(p.id).padStart(2, "0")}</span>
          <span class="ex-coord">${formatLatLng(p.lat, p.lng)}</span>
          <span class="ex-remove" aria-hidden="true">×</span>
        </button>
      `;
      e.positiveList.appendChild(li);
    }
    e.positiveList.querySelectorAll<HTMLButtonElement>("button[data-pid]").forEach((b) => {
      b.addEventListener("click", (ev) => {
        const target = ev.target as HTMLElement;
        const pid = Number(b.dataset.pid);
        if (target.closest(".ex-remove")) {
          state.positivePoints = state.positivePoints
            .filter((p) => p.id !== pid)
            .map((p, idx) => ({ ...p, id: idx + 1 }));
          globe.setPositives(state.positivePoints);
          void scoreCandidates();
          updateView();
        } else {
          const pt = state.positivePoints.find((p) => p.id === pid);
          if (pt) globe.map.flyTo({ center: [pt.lng, pt.lat], zoom: 13 });
        }
      });
    });
  }

  // Results list — pick the right source and slice for the active view
  const activeResults =
    state.viewMode === "outlier" ? state.outlierResults
    : state.viewMode === "surprise" ? state.surpriseResults
    : state.viewMode === "gradient" ? state.gradientResults
    : state.results;
  const visible = state.viewMode === "topk"
    ? activeResults.slice(0, state.topK)
    : state.viewMode === "threshold"
      ? activeResults.filter((r) => r.score <= state.threshold)
      : activeResults.slice(0, state.topK);

  if (e.resultCount) {
    const needsExemplars = state.viewMode !== "outlier" && state.viewMode !== "surprise";
    if (!state.candidateRows.length) e.resultCount.textContent = "";
    else if (needsExemplars && !state.positivePoints.length) e.resultCount.textContent = "";
    else if (state.viewMode === "threshold")
      e.resultCount.textContent = `${compactNum(visible.length)} / ${compactNum(activeResults.length)} cutoff`;
    else if (state.viewMode === "topk")
      e.resultCount.textContent = `${visible.length} / ${compactNum(activeResults.length)}`;
    else
      e.resultCount.textContent = `${compactNum(activeResults.length)} scored`;
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
      b.addEventListener("click", () => globe.flyToBBox(row.bbox, { zoom: 13 }));
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
  const centroids = new Float64Array(candidates.length * 2);
  for (let i = 0; i < candidates.length; i++) {
    const c = centroid(candidates[i].bbox);
    centroids[i * 2] = c.lat;
    centroids[i * 2 + 1] = c.lng;
  }
  worker.postMessage({
    type: "init",
    embeddings: candidates.map((c) => new Uint8Array(c.embedding)),
    centroids,
  });
  scoringWorkerReady = true;
}

function combineEmbeddings(embeddings: Uint8Array[], method: CombineMethod): Uint8Array[] {
  if (embeddings.length <= 1 || method === "mean") return embeddings;
  const len = embeddings[0].length;
  const result = new Uint8Array(len);
  if (method === "and") {
    result.set(embeddings[0]);
    for (let e = 1; e < embeddings.length; e++) {
      for (let i = 0; i < len; i++) result[i] &= embeddings[e][i];
    }
  } else if (method === "or") {
    for (let e = 0; e < embeddings.length; e++) {
      for (let i = 0; i < len; i++) result[i] |= embeddings[e][i];
    }
  } else if (method === "xor") {
    for (let e = 0; e < embeddings.length; e++) {
      for (let i = 0; i < len; i++) result[i] ^= embeddings[e][i];
    }
  }
  return [result];
}

function maybeInvertEmbedding(emb: Uint8Array): Uint8Array {
  if (!state.invertSearch) return emb;
  const inv = new Uint8Array(emb.length);
  for (let i = 0; i < emb.length; i++) inv[i] = emb[i] ^ 0xFF;
  return inv;
}

function resolveNegativeEmbeddings(): Uint8Array[] {
  return state.negativePoints.flatMap((point) => {
    if (point.embedding) return [maybeInvertEmbedding(point.embedding)];
    const intersecting = state.candidateRows.filter((c) => containsPoint(c.bbox, point.lat, point.lng));
    if (!intersecting.length) return [];
    let best = intersecting[0];
    let bestD = Number.POSITIVE_INFINITY;
    for (const option of intersecting) {
      const c = centroid(option.bbox);
      const d = distanceSquared(point.lat, point.lng, c.lat, c.lng);
      if (d < bestD) { bestD = d; best = option; }
    }
    return [maybeInvertEmbedding(best.embedding)];
  });
}

async function scoreWithWorker(exemplars: CandidateRow[]): Promise<RankedRow[]> {
  if (!scoringWorkerReady) return [];
  const worker = ensureScoringWorker();
  const requestId = ++scoringRequestId;
  const excludeIndices = new Set(exemplars.map((ex) => state.candidateRows.indexOf(ex)).filter((i) => i >= 0));
  let posEmbeddings = exemplars.map((ex) => maybeInvertEmbedding(new Uint8Array(ex.embedding)));
  posEmbeddings = combineEmbeddings(posEmbeddings, state.combineMethod);
  const negEmbeddings = resolveNegativeEmbeddings();
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
      exemplars: posEmbeddings,
      negatives: negEmbeddings,
      excludeIndices: [...excludeIndices],
    });
  });
  return results.map(({ index, score }) => ({ ...state.candidateRows[index], score }));
}

function resolvePositiveMatches(): PositiveMatch[] {
  return state.positivePoints.flatMap((point) => {
    // External exemplar — already has its embedding
    if (point.embedding && point.chips_id) {
      return [{
        pointId: point.id,
        candidate: {
          chips_id: point.chips_id,
          bbox: { west: point.lng - 0.005, south: point.lat - 0.005, east: point.lng + 0.005, north: point.lat + 0.005 },
          embedding: point.embedding,
          shard_path: "",
        },
      }];
    }
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

/* ---------------------------------------------------------- Overlay toggle */

function activeResultsForMode(): RankedRow[] {
  if (state.viewMode === "outlier") return state.outlierResults;
  if (state.viewMode === "surprise") return state.surpriseResults;
  if (state.viewMode === "gradient") return state.gradientResults;
  return state.results;
}

function applyOverlay(): void {
  if (state.overlayVisible) {
    const activeResults = activeResultsForMode();
    if (state.viewMode === "threshold") {
      const filtered = activeResults.filter((r) => r.score <= state.threshold);
      globe.setResults(filtered, filtered.length, state.viewMode);
    } else {
      globe.setResults(activeResults, state.topK, state.viewMode);
    }
  } else {
    globe.setResults([], 0, state.viewMode);
  }
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

/* ---------------------------------------------------------- Surprise scoring */

async function computeSurprise(): Promise<void> {
  if (!scoringWorkerReady || !state.candidateRows.length) return;
  if (state.surpriseComputed) return;

  setStatus("Computing spatial surprise scores\u2026");
  const worker = ensureScoringWorker();
  const requestId = ++scoringRequestId;
  const results = await new Promise<WorkerScoreResult[]>((resolve, reject) => {
    const onMessage = (event: MessageEvent<{ type: string; requestId: number; results: WorkerScoreResult[] }>) => {
      if (event.data.type !== "surprise-result" || event.data.requestId !== requestId) return;
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
    worker.postMessage({ type: "surprise", requestId, k: 8 });
  });

  state.surpriseResults = results.map(({ index, score }) => ({ ...state.candidateRows[index], score }));
  state.surpriseComputed = true;
  globe.setResults(state.surpriseResults, state.topK, "surprise");
  setStatus(`Surprise analysis: ${state.surpriseResults.length} patches scored. Brightest = most surprising.`);
  updateView();
}

/* ---------------------------------------------------------- Gradient scoring */

async function computeGradient(): Promise<void> {
  if (!scoringWorkerReady || !state.candidateRows.length || !state.results.length) return;

  setStatus("Computing similarity gradient\u2026");
  const worker = ensureScoringWorker();
  const requestId = ++scoringRequestId;
  const scoreArr = new Float64Array(state.candidateRows.length);
  const scoreMap = new Map<string, number>();
  for (const r of state.results) scoreMap.set(r.chips_id, r.score);
  for (let i = 0; i < state.candidateRows.length; i++) {
    scoreArr[i] = scoreMap.get(state.candidateRows[i].chips_id) ?? 0;
  }
  const results = await new Promise<WorkerScoreResult[]>((resolve, reject) => {
    const onMessage = (event: MessageEvent<{ type: string; requestId: number; results: WorkerScoreResult[] }>) => {
      if (event.data.type !== "gradient-result" || event.data.requestId !== requestId) return;
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
    worker.postMessage({ type: "gradient", requestId, scores: scoreArr, k: 6 });
  });

  state.gradientResults = results.map(({ index, score }) => ({ ...state.candidateRows[index], score }));
  globe.setResults(state.gradientResults, state.topK, "gradient");
  setStatus(`Gradient analysis: ${state.gradientResults.length} patches scored. Brightest = strongest boundary.`);
  updateView();
}

/* ---------------------------------------------------------- Region fingerprint */

async function regionFingerprint(): Promise<void> {
  if (!state.positivePoints.length || !state.candidateRows.length) return;

  const e = els();
  setStatus("Computing region fingerprint\u2026");

  // 1. Compute fingerprint: majority-vote binary embedding from positives
  const matches = resolvePositiveMatches();
  const exemplarEmbs = matches.map((m) => m.candidate.embedding);
  if (!exemplarEmbs.length) { setStatus("No matched exemplar patches."); return; }
  const len = exemplarEmbs[0].length;
  const fingerprint = new Uint8Array(len);
  for (let byteIdx = 0; byteIdx < len; byteIdx++) {
    let bits = 0;
    for (let bit = 0; bit < 8; bit++) {
      let ones = 0;
      for (const emb of exemplarEmbs) {
        if ((emb[byteIdx] >> bit) & 1) ones++;
      }
      if (ones > exemplarEmbs.length / 2) bits |= (1 << bit);
    }
    fingerprint[byteIdx] = bits;
  }

  // 2. Sample up to 20 random shards from manifest (exclude current)
  try {
    const db = await getDuckDB();
    const conn = await db.connect();
    const allManifest = await conn.query(`
      SELECT path, rows, xmin, ymin, xmax, ymax
      FROM read_parquet(${sqlString(MANIFEST_URL)})
      ORDER BY path
    `.trim());
    const allShards = allManifest.toArray() as ManifestRow[];
    await conn.close();

    const currentPaths = new Set(state.manifestShards.map((s) => s.path));
    const candidates = allShards.filter((s) => !currentPaths.has(s.path));
    // Shuffle and take 20
    for (let i = candidates.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
    }
    const sampled = candidates.slice(0, 20);

    if (!sampled.length) { setStatus("No other shards to compare."); return; }

    // 3. For each shard, fetch a few patches and compute mean hamming
    type ShardResult = { shard: ManifestRow; meanDist: number };
    const shardResults: ShardResult[] = [];

    for (let si = 0; si < sampled.length; si++) {
      const shard = sampled[si];
      setStatus(`Fingerprinting region ${si + 1}/${sampled.length}\u2026`);
      try {
        const url = resolveShardUrl(shard.path);
        const conn2 = await db.connect();
        const result = await conn2.query(`
          SELECT chips_id, bbox, embedding
          FROM read_parquet(${sqlString(url)})
          LIMIT 50
        `.trim());
        const rows = result.toArray() as Array<{
          chips_id: string;
          bbox: { xmin: number; ymin: number; xmax: number; ymax: number };
          embedding: unknown;
        }>;
        await conn2.close();
        if (!rows.length) continue;
        let totalDist = 0;
        for (const row of rows) {
          const emb = normalizeEmbedding(row.embedding);
          let d = 0;
          for (let i = 0; i < len; i++) {
            let x = fingerprint[i] ^ emb[i];
            while (x) { d += x & 1; x >>= 1; }
          }
          totalDist += d;
        }
        shardResults.push({ shard, meanDist: totalDist / rows.length });
      } catch {
        // skip failed shards
      }
    }

    shardResults.sort((a, b) => a.meanDist - b.meanDist);

    // 4. Render results
    if (e.fingerprintResults) {
      e.fingerprintResults.hidden = false;
      e.fingerprintResults.innerHTML = `
        <div class="fp-header">Similar regions <button class="fp-close" type="button">\u00d7</button></div>
        <ul class="fp-list">
          ${shardResults.slice(0, 10).map((sr, i) => {
            const cx = ((sr.shard.xmin + sr.shard.xmax) / 2).toFixed(1);
            const cy = ((sr.shard.ymin + sr.shard.ymax) / 2).toFixed(1);
            return `<li><button type="button" data-fp="${i}" class="fp-item">
              <span class="rank">${String(i + 1).padStart(2, "0")}</span>
              <span class="rank-body">
                <span class="rank-coord">${cy}\u00b0, ${cx}\u00b0</span>
                <span class="rank-chip">${sr.shard.path.split("/").pop()}</span>
              </span>
              <span class="rank-score">${sr.meanDist.toFixed(1)}</span>
            </button></li>`;
          }).join("")}
        </ul>`;

      e.fingerprintResults.querySelector(".fp-close")?.addEventListener("click", () => {
        if (e.fingerprintResults) e.fingerprintResults.hidden = true;
      });
      e.fingerprintResults.querySelectorAll<HTMLButtonElement>("button[data-fp]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const idx = Number(btn.dataset.fp);
          const sr = shardResults[idx];
          const bbox: BBox = { west: sr.shard.xmin, south: sr.shard.ymin, east: sr.shard.xmax, north: sr.shard.ymax };
          globe.fitBounds(bbox, { padding: 60, maxZoom: 11 });
        });
      });
    }

    setStatus(`Found ${shardResults.length} similar regions.`);
  } catch (err) {
    setStatus(`Fingerprint failed: ${err instanceof Error ? err.message : String(err)}`);
  }
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
  // Preserve external exemplars (those with their own embedding) across region changes.
  state.bbox = bbox;
  const externalExemplars = state.positivePoints.filter((p) => p.embedding);
  state.positivePoints = externalExemplars.map((p, i) => ({ ...p, id: i + 1 }));
  state.negativePoints = [];
  state.positiveMatches = [];
  state.results = [];
  state.outlierResults = [];
  state.outlierComputed = false;
  state.surpriseResults = [];
  state.surpriseComputed = false;
  state.gradientResults = [];
  state.threshold = Infinity;
  state.candidateRows = [];
  state.manifestShards = [];
  state.loading = true;
  globe.setAoi(bbox);
  globe.setPositives(state.positivePoints);
  globe.setNegatives(state.negativePoints);
  globe.setPositiveMatches([]);
  globe.setResults([], state.topK, state.viewMode);
  globe.setPreview(null);
  setStatus("Fetching intersecting shards\u2026");
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
      ? `Region loaded — ${new Intl.NumberFormat().format(state.candidateRows.length)} patches from ${shards.length - failed.length}/${shards.length} shards. Click anywhere to seed an exemplar.`
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

async function fetchExternalEmbedding(lat: number, lng: number): Promise<CandidateRow | null> {
  const db = await getDuckDB();
  const conn = await db.connect();
  try {
    // Find which shard(s) contain this point
    const mResult = await conn.query(`
      SELECT path FROM read_parquet(${sqlString(MANIFEST_URL)})
      WHERE xmin <= ${lng} AND xmax >= ${lng}
        AND ymin <= ${lat} AND ymax >= ${lat}
      ORDER BY rows ASC LIMIT 5
    `.trim());
    const shards = mResult.toArray() as Array<{ path: string }>;
    if (!shards.length) return null;

    for (const shard of shards) {
      const url = resolveShardUrl(shard.path);
      const rows = await fetchShardCandidates(db, url, {
        west: lng - 0.01, south: lat - 0.01,
        east: lng + 0.01, north: lat + 0.01,
      });
      const hit = rows.find((r) => containsPoint(r.bbox, lat, lng));
      if (hit) return hit;
    }
    return null;
  } finally {
    await conn.close();
  }
}

function addPositive(lat: number, lng: number): void {
  const isInsideAoi = state.bbox && containsPoint(state.bbox, lat, lng) && state.candidateRows.length;

  if (isInsideAoi) {
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
  } else {
    // External exemplar — fetch embedding from remote parquet
    setStatus("Fetching external exemplar embedding…");
    updateView();
    void fetchExternalEmbedding(lat, lng).then((row) => {
      if (!row) {
        setStatus("No patch found at that location.");
        return;
      }
      // Deduplicate by chips_id
      const existing = state.positivePoints.some((p) => p.embedding && p.chips_id === row.chips_id);
      if (existing) {
        setStatus("That patch is already selected.");
        return;
      }
      state.positivePoints.push({
        id: state.positivePoints.length + 1,
        lat, lng,
        embedding: row.embedding,
        chips_id: row.chips_id,
      });
      globe.setPositives(state.positivePoints);
      void scoreCandidates();
      updateView();
    });
  }
}

function addNegative(lat: number, lng: number): void {
  if (!state.bbox || !state.candidateRows.length) return;
  const isInsideAoi = containsPoint(state.bbox, lat, lng);
  if (!isInsideAoi) return;

  const patch = findNearestCandidate(lat, lng);
  if (patch) {
    const existing = state.negativePoints.some((p) => {
      const ep = findNearestCandidate(p.lat, p.lng);
      return ep && ep.chips_id === patch.chips_id;
    });
    if (existing) return;
  }
  state.negativePoints.push({ id: state.negativePoints.length + 1, lat, lng });
  globe.setNegatives(state.negativePoints);
  void scoreCandidates();
  updateView();
}

function clearNegatives(): void {
  if (!state.negativePoints.length) return;
  state.negativePoints = [];
  globe.setNegatives([]);
  void scoreCandidates();
  updateView();
}

function clearPoints(): void {
  if (!state.positivePoints.length && !state.negativePoints.length) return;
  state.positivePoints = [];
  state.negativePoints = [];
  state.positiveMatches = [];
  state.results = [];
  state.gradientResults = [];
  globe.setPositives([]);
  globe.setNegatives([]);
  globe.setPositiveMatches([]);
  globe.setResults([], state.topK, state.viewMode);
  globe.setPreview(null);
  setStatus("Exemplar points cleared. Click anywhere to seed new ones.");
  updateView();
}

function clearRegion(): void {
  state.bbox = null;
  state.manifestShards = [];
  state.candidateRows = [];
  state.positivePoints = [];
  state.negativePoints = [];
  state.positiveMatches = [];
  state.results = [];
  state.topK = DEFAULT_TOP_K;
  state.viewMode = "topk";
  state.outlierResults = [];
  state.outlierComputed = false;
  state.surpriseResults = [];
  state.surpriseComputed = false;
  state.gradientResults = [];
  state.threshold = Infinity;
  state.invertSearch = false;
  state.combineMethod = "mean";
  globe.setAoi(null);
  globe.setPositives([]);
  globe.setNegatives([]);
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
  e.overlayToggle?.addEventListener("click", () => {
    state.overlayVisible = !state.overlayVisible;
    applyOverlay();
    updateView();
  });
  // Body-level help popover — escapes all overflow/transform containing blocks
  const helpContent: Record<string, string> = {
    retrieval: `<p class="overlay-help-title">Overlay modes</p><ul class="overlay-help-list"><li><strong>Top-K</strong> — Ranked list of the N most similar patches to your query.</li><li><strong>Heat</strong> — Similarity heatmap across all patches; bright = close match.</li><li><strong>Cutoff</strong> — All patches within a max Hamming distance you set.</li><li><strong>Outlier</strong> — Patches most unlike the rest of the visible region.</li><li><strong>Surprise</strong> — Patches that differ sharply from their spatial neighbors.</li><li><strong>Edge</strong> — Boundaries where similarity scores change abruptly.</li></ul>`,
    explore: `<p class="overlay-help-title">Explore</p><p class="panel-info-body">Curated regions based on prior knowledge — interesting places around the world to search within.</p>`,
    discoveries: `<p class="overlay-help-title">Discoveries</p><p class="panel-info-body">Interesting locations surfaced automatically by analyzing the embeddings — outliers, surprising patches, and boundary regions found without manual curation.</p>`,
  };
  const helpPopover = document.createElement("div");
  helpPopover.className = "help-popover";
  helpPopover.setAttribute("role", "tooltip");
  document.body.appendChild(helpPopover);
  document.querySelectorAll<HTMLButtonElement>("button[data-help]").forEach((btn) => {
    btn.addEventListener("mouseenter", () => {
      const key = btn.dataset.help ?? "";
      if (!helpContent[key]) return;
      helpPopover.innerHTML = helpContent[key];
      const r = btn.getBoundingClientRect();
      helpPopover.style.top = `${r.bottom + 8}px`;
      helpPopover.style.right = `${window.innerWidth - r.right}px`;
      helpPopover.style.display = "block";
    });
    btn.addEventListener("mouseleave", () => {
      helpPopover.style.display = "none";
    });
  });
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
      } else if (mode === "surprise" && !state.surpriseComputed) {
        void computeSurprise();
      } else if (mode === "surprise") {
        globe.setResults(state.surpriseResults, state.topK, mode);
      } else if (mode === "gradient") {
        if (state.results.length) {
          void computeGradient();
        }
      } else if (mode === "threshold") {
        const filtered = state.results.filter((r) => r.score <= state.threshold);
        globe.setResults(filtered, filtered.length, mode);
      } else {
        globe.setResults(state.results, state.topK, mode);
      }
      updateView();
    });
  });

  // Invert toggle
  e.invertToggle?.addEventListener("click", () => {
    state.invertSearch = !state.invertSearch;
    void scoreCandidates();
    updateView();
  });

  // Combine method
  e.combineSelect?.addEventListener("change", (ev) => {
    state.combineMethod = (ev.currentTarget as HTMLSelectElement).value as CombineMethod;
    void scoreCandidates();
    updateView();
  });

  // Clear negatives
  e.clearNegativesBtn?.addEventListener("click", clearNegatives);

  // Fingerprint
  e.fingerprintBtn?.addEventListener("click", () => void regionFingerprint());

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
    onNegativeClick: (lat, lng) => addNegative(lat, lng),
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
  renderInterestingPoints();
  updateView();
  wireTutorial();
}

/* ================================================================ Tutorial */

type TutorialPlacement = "top" | "bottom" | "left" | "right" | "center";

type TutorialStep = {
  target?: string;         // CSS selector; omit for centered card
  title: string;
  body: string;
  placement?: TutorialPlacement;
  padding?: number;        // extra glow padding around spotlight (px)
};

const TUTORIAL_STEPS: TutorialStep[] = [
  {
    title: "Welcome to terrabit",
    body: "This tour walks you through searching Earth's surface using binary embeddings — a fast, compact way to find similar satellite patches.",
    placement: "center",
  },
  {
    target: "#search-wrap",
    title: "Search anywhere",
    body: "Type a city, park, country, or paste coordinates. The globe flies to your destination and you can define a region from there.",
    placement: "bottom",
    padding: 10,
  },
  {
    target: "#draw-btn",
    title: "Draw a region",
    body: "Click <strong>Draw region</strong> then drag on the map, or hold <kbd>Shift</kbd> and drag anywhere on the globe to define your area of interest (AOI).",
    placement: "right",
    padding: 12,
  },
  {
    target: "#aoi-nav",
    title: "Preset regions",
    body: "No need to draw — pick a preset AOI from this panel. Locations range from coral atolls to arctic glaciers, solar farms, and megacities.",
    placement: "left",
    padding: 10,
  },
  {
    target: "#positive-list",
    title: "Exemplar points",
    body: "After loading a region, <strong>click anywhere on the map</strong> to place a positive exemplar. terrabit finds all patches with similar binary embeddings.",
    placement: "right",
    padding: 8,
  },
  {
    target: "#negative-section",
    title: "Negative exemplars",
    body: "<strong>Right-click</strong> or <strong>Shift+click</strong> on the map to add negative exemplars. These actively push results <em>away</em> from unwanted features.",
    placement: "right",
    padding: 8,
  },
  {
    target: ".view-toggle",
    title: "View modes",
    body: "Switch how results are displayed: <strong>Top-K</strong> ranks the best matches, <strong>Heat</strong> maps similarity across the AOI, <strong>Outlier</strong> surfaces unique patches, <strong>Surprise</strong> finds spatially unexpected tiles, <strong>Edge</strong> detects similarity boundaries, and <strong>Cutoff</strong> lets you set a distance threshold.",
    placement: "right",
    padding: 10,
  },
  {
    target: "#combine-method",
    title: "Combine method",
    body: "Using multiple exemplars? Choose how their embeddings are merged: <strong>Mean</strong> averages them, <strong>AND</strong>/<strong>OR</strong>/<strong>XOR</strong> apply bitwise logic for more precise control.",
    placement: "right",
    padding: 10,
  },
  {
    target: "#invert-toggle",
    title: "Invert search",
    body: "Toggle <strong>Invert</strong> to flip the query — finds patches that are the <em>opposite</em> of your exemplars. Great for contrast searches.",
    placement: "right",
    padding: 12,
  },
  {
    target: "#fingerprint-btn",
    title: "Region fingerprint",
    body: "Click <strong>Find similar regions</strong> to compute a fingerprint for the entire AOI and discover other places on the globe that look like it.",
    placement: "top",
    padding: 12,
  },
  {
    target: "#export-btn",
    title: "Export results",
    body: "Download your ranked results as <strong>GeoParquet</strong> — ready for analysis in QGIS, DuckDB, GeoPandas, or any geo toolchain.",
    placement: "top",
    padding: 12,
  },
  {
    title: "You're ready to explore",
    body: "Spin the globe, draw a region, drop some exemplars, and let binary embeddings do the rest. Happy searching.",
    placement: "center",
  },
];

const TUTORIAL_STORAGE_KEY = "terrabit_tutorial_seen_v1";

type TutorialState = {
  active: boolean;
  step: number;
};

const tutorialState: TutorialState = { active: false, step: 0 };

let tutorialOverlay: HTMLElement | null = null;
let tutorialCard: HTMLElement | null = null;
let tutorialRaf: number | null = null;

function tutorialGetEl(): { overlay: HTMLElement; card: HTMLElement } {
  if (!tutorialOverlay) {
    tutorialOverlay = document.createElement("div");
    tutorialOverlay.className = "tut-overlay";
    tutorialOverlay.id = "tut-overlay";
    document.body.appendChild(tutorialOverlay);
  }
  if (!tutorialCard) {
    tutorialCard = document.createElement("div");
    tutorialCard.className = "tut-card";
    tutorialCard.id = "tut-card";
    document.body.appendChild(tutorialCard);
  }
  return { overlay: tutorialOverlay, card: tutorialCard };
}

function tutorialPositionCard(
  card: HTMLElement,
  target: Element | null,
  placement: TutorialPlacement,
  padding: number,
): void {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const cw = card.offsetWidth || 320;
  const ch = card.offsetHeight || 200;
  const margin = 18;

  if (!target || placement === "center") {
    card.style.left = `${(vw - cw) / 2}px`;
    card.style.top = `${(vh - ch) / 2}px`;
    return;
  }

  const r = target.getBoundingClientRect();
  const pad = padding;

  let left = 0;
  let top = 0;

  if (placement === "right") {
    left = r.right + pad + margin;
    top = r.top + r.height / 2 - ch / 2;
  } else if (placement === "left") {
    left = r.left - pad - cw - margin;
    top = r.top + r.height / 2 - ch / 2;
  } else if (placement === "bottom") {
    left = r.left + r.width / 2 - cw / 2;
    top = r.bottom + pad + margin;
  } else {
    // top
    left = r.left + r.width / 2 - cw / 2;
    top = r.top - pad - ch - margin;
  }

  // Clamp to viewport
  left = Math.max(margin, Math.min(left, vw - cw - margin));
  top = Math.max(margin, Math.min(top, vh - ch - margin));

  card.style.left = `${left}px`;
  card.style.top = `${top}px`;
}

function tutorialPaintShadowOverlay(overlay: HTMLElement, target: Element | null, padding: number): void {
  if (!target) {
    // Full dark overlay, no cutout
    overlay.style.removeProperty("background");
    overlay.style.setProperty("--tut-shadow-inset", "none");
    overlay.classList.remove("has-spotlight");
    return;
  }
  const r = target.getBoundingClientRect();
  const pad = padding;
  const rx = Math.max(0, r.left - pad);
  const ry = Math.max(0, r.top - pad);
  const rw = r.width + pad * 2;
  const rh = r.height + pad * 2;
  // Store as CSS custom props and handle via JS-driven box-shadow on the spotlight hole div
  overlay.style.setProperty("--tut-spot-x", `${rx}px`);
  overlay.style.setProperty("--tut-spot-y", `${ry}px`);
  overlay.style.setProperty("--tut-spot-w", `${rw}px`);
  overlay.style.setProperty("--tut-spot-h", `${rh}px`);
  overlay.classList.add("has-spotlight");
}

function tutorialRender(): void {
  const { overlay, card } = tutorialGetEl();
  const step = TUTORIAL_STEPS[tutorialState.step];
  if (!step) return;

  const target = step.target ? document.querySelector(step.target) : null;
  const placement = step.placement ?? "bottom";
  const padding = step.padding ?? 10;
  const isCenter = placement === "center" || !target;
  const total = TUTORIAL_STEPS.length;
  const idx = tutorialState.step;

  // Paint overlay + spotlight
  tutorialPaintShadowOverlay(overlay, isCenter ? null : target, padding);

  // Build card HTML
  const dots = Array.from({ length: total }, (_, i) =>
    `<span class="tut-dot${i === idx ? " is-active" : ""}"></span>`
  ).join("");

  card.innerHTML = `
    <button class="tut-dismiss" aria-label="Close tutorial" id="tut-close">✕</button>
    <p class="tut-step-label">${idx + 1} / ${total}</p>
    <h3 class="tut-title">${step.title}</h3>
    <p class="tut-body">${step.body}</p>
    <div class="tut-dots">${dots}</div>
    <div class="tut-actions">
      <button class="tut-btn tut-btn-ghost" id="tut-skip" type="button">Skip tour</button>
      <div class="tut-nav">
        ${idx > 0 ? `<button class="tut-btn tut-btn-secondary" id="tut-back" type="button">← Back</button>` : ""}
        ${idx < total - 1
          ? `<button class="tut-btn tut-btn-primary" id="tut-next" type="button">Next →</button>`
          : `<button class="tut-btn tut-btn-primary" id="tut-done" type="button">Get started</button>`}
      </div>
    </div>
  `;

  // Force a layout read so the browser registers the initial state before transitioning
  void overlay.offsetWidth;
  void card.offsetWidth;

  // Make visible
  overlay.classList.add("is-active");
  card.classList.add("is-active");

  // Position card after paint (need dimensions)
  requestAnimationFrame(() => {
    tutorialPositionCard(card, isCenter ? null : target, placement, padding);
  });

  // Wire card buttons
  card.querySelector("#tut-close")?.addEventListener("click", tutorialStop);
  card.querySelector("#tut-skip")?.addEventListener("click", tutorialStop);
  card.querySelector("#tut-back")?.addEventListener("click", () => tutorialGo(idx - 1));
  card.querySelector("#tut-next")?.addEventListener("click", () => tutorialGo(idx + 1));
  card.querySelector("#tut-done")?.addEventListener("click", tutorialStop);
}

function tutorialGo(step: number): void {
  tutorialState.step = Math.max(0, Math.min(step, TUTORIAL_STEPS.length - 1));
  // Fade out card, then re-render
  const card = document.querySelector<HTMLElement>("#tut-card");
  const overlay = document.querySelector<HTMLElement>("#tut-overlay");
  if (card) {
    card.classList.add("is-transitioning");
    if (overlay) overlay.classList.add("is-transitioning");
    if (tutorialRaf) cancelAnimationFrame(tutorialRaf);
    tutorialRaf = requestAnimationFrame(() => {
      tutorialRaf = requestAnimationFrame(() => {
        card.classList.remove("is-transitioning");
        if (overlay) overlay.classList.remove("is-transitioning");
        tutorialRender();
      });
    });
  } else {
    tutorialRender();
  }
}

function tutorialStart(): void {
  tutorialState.active = true;
  tutorialState.step = 0;
  tutorialRender();
}

function tutorialStop(): void {
  tutorialState.active = false;
  const overlay = document.querySelector<HTMLElement>("#tut-overlay");
  const card = document.querySelector<HTMLElement>("#tut-card");
  if (overlay) overlay.classList.remove("is-active", "has-spotlight", "is-transitioning");
  if (card) card.classList.remove("is-active", "is-transitioning");
}

function wireTutorial(): void {
  const btn = document.querySelector<HTMLButtonElement>("#tutorial-trigger");
  btn?.addEventListener("click", (ev) => {
    ev.stopPropagation();
    if (tutorialState.active) tutorialStop();
    else tutorialStart();
  });

  // Always show tutorial on start
  setTimeout(tutorialStart, 900);

  // Esc to close, Enter to advance
  window.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && tutorialState.active) tutorialStop();
    if (ev.key === "Enter" && tutorialState.active) {
      const next = tutorialState.step + 1;
      if (next < TUTORIAL_STEPS.length) tutorialGo(next);
      else tutorialStop();
    }
  });

  // Click overlay to advance / close
  document.addEventListener("click", (ev) => {
    if (!tutorialState.active) return;
    const overlay = document.querySelector<HTMLElement>("#tut-overlay");
    const card = document.querySelector<HTMLElement>("#tut-card");
    const t = ev.target as Node;
    if (overlay && overlay === t) {
      // click on dark area → advance or close
      const next = tutorialState.step + 1;
      if (next < TUTORIAL_STEPS.length) tutorialGo(next);
      else tutorialStop();
    }
    // ignore clicks inside card (handled by card buttons)
    if (card && card.contains(t)) return;
  });
}

bootstrap();
