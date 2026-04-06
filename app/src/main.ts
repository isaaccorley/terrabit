import * as duckdb from "@duckdb/duckdb-wasm";
import duckdbWasmEh from "@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url";
import duckdbWasmMvp from "@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url";
import duckdbWorkerEh from "@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url";
import duckdbWorkerMvp from "@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url";

import "./styles.css";
import { GlobeMap } from "./map";
import type {
  AoiEntry,
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
  pointInPolygon,
} from "./util";

const MANIFEST_URL =
  "https://data.source.coop/geospatialml/terrabit/clay-v1_5-binary-sentinel-2/manifest.parquet";
const DEFAULT_TOP_K = 50;
const MAX_TOP_K = 100;

type AppState = {
  bboxes: AoiEntry[];
  nextAoiId: number;
  regionRows: Map<number, CandidateRow[]>;
  regionShardCounts: Map<number, number>;
  status: string;
  candidateRows: CandidateRow[];
  positivePoints: PositivePoint[];
  negativePoints: NegativePoint[];
  positiveMatches: PositiveMatch[];
  baseResults: RankedRow[];
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
  bboxes: [],
  nextAoiId: 1,
  regionRows: new Map(),
  regionShardCounts: new Map(),
  status: "Spin the globe. Zoom. Shift-drag or hit Draw region to define an AOI.",
  candidateRows: [],
  positivePoints: [],
  negativePoints: [],
  positiveMatches: [],
  baseResults: [],
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
const regionLoadRunIds = new Map<number, number>();

// In-flight guards so background precompute and on-click don't double-fire
let outlierComputing = false;
let surpriseComputing = false;
let gradientComputing = false;

function resetComputeState(): void {
  outlierComputing = false;
  surpriseComputing = false;
  gradientComputing = false;
}

function isInsideAnyAoi(lat: number, lng: number): boolean {
  return state.bboxes.some((e) =>
    e.polygon ? pointInPolygon(e.polygon, lat, lng) : containsPoint(e.bbox, lat, lng)
  );
}

// List render fingerprints — skip DOM rebuild when data hasn't changed
let lastPositiveListKey = "";
let lastNegativeListKey = "";
let lastResultListKey = "";

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
          <span class="panel-kicker">Query</span>
        </header>
        <div class="draw-row">
          <button id="draw-btn" class="btn btn-sm btn-primary" type="button">
            <span class="btn-glyph">▢</span>
            <span id="draw-label">Draw region</span>
          </button>
          <button id="draw-poly-btn" class="btn btn-sm btn-ghost" type="button" title="Draw polygon region — click to add vertices, double-click to close">
            <span class="btn-glyph">⬡</span>
            <span>Polygon</span>
          </button>
          <button id="zoom-region-btn" class="icon-btn" type="button" hidden title="Zoom to region(s)">
            <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M6 2H2v4"/><path d="M14 6V2h-4"/><path d="M2 10v4h4"/><path d="M10 14h4v-4"/></svg>
          </button>
        </div>
        <div id="active-regions" class="active-regions"></div>
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
                <span class="panel-kicker">Exemplars</span>
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
                <span class="panel-kicker">Retrieval</span>

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
            <div class="action-row">
              <button id="export-btn" class="btn btn-sm btn-ghost action-btn" type="button" hidden>
                <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M2 11v3h12v-3"/><path d="M8 2v8"/><path d="M5 7l3 3 3-3"/></svg>
                <span>Export</span>
              </button>
            </div>
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
    drawPolyBtn: document.querySelector<HTMLButtonElement>("#draw-poly-btn"),
    zoomRegionBtn: document.querySelector<HTMLButtonElement>("#zoom-region-btn"),
    activeRegions: document.querySelector<HTMLDivElement>("#active-regions"),
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
  const polyArmed = armed && globe?.getDrawMode() === "polygon";
  const rectArmed = armed && !polyArmed;
  e.drawLabel.textContent = rectArmed ? "Drawing…" : "Draw region";
  e.drawBtn.classList.toggle("is-armed", rectArmed);
  e.drawPolyBtn?.classList.toggle("is-armed", polyArmed);

  const totalShards = [...state.regionShardCounts.values()].reduce((a, b) => a + b, 0);
  if (e.mShards) e.mShards.textContent = totalShards ? String(totalShards) : "—";
  if (e.mPatches) e.mPatches.textContent = state.candidateRows.length ? new Intl.NumberFormat().format(state.candidateRows.length) : "—";
  if (e.mRoi) {
    if (!state.bboxes.length) e.mRoi.textContent = "—";
    else if (state.bboxes.length === 1) {
      const b = state.bboxes[0].bbox;
      e.mRoi.textContent = `${(b.east - b.west).toFixed(2)}°×${(b.north - b.south).toFixed(2)}°`;
    } else {
      e.mRoi.textContent = `${state.bboxes.length} regions`;
    }
  }
  if (e.zoomRegionBtn) e.zoomRegionBtn.hidden = state.bboxes.length === 0;

  // Active-regions chips
  if (e.activeRegions) {
    const regKey = state.bboxes.map((a) => a.id).join(",");
    if (e.activeRegions.dataset.key !== regKey) {
      e.activeRegions.dataset.key = regKey;
      e.activeRegions.innerHTML = "";
      if (state.bboxes.length) {
        for (const entry of state.bboxes) {
          const chip = document.createElement("div");
          chip.className = "aoi-chip";
          chip.innerHTML = `<span class="aoi-chip-label">AOI ${entry.id}</span><button class="aoi-chip-remove" data-aoi-id="${entry.id}" title="Remove region" aria-label="Remove AOI ${entry.id}">×</button>`;
          e.activeRegions.appendChild(chip);
        }
        const clearAll = document.createElement("button");
        clearAll.className = "clear-all-btn";
        clearAll.id = "clear-all-regions-btn";
        clearAll.textContent = "Clear all";
        e.activeRegions.appendChild(clearAll);
        e.activeRegions.querySelectorAll<HTMLButtonElement>(".aoi-chip-remove").forEach((btn) => {
          btn.addEventListener("click", () => {
            const id = Number(btn.dataset.aoiId);
            removeRegion(id);
          });
        });
        document.getElementById("clear-all-regions-btn")?.addEventListener("click", clearAllRegions);
      }
    }
  }

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
  e.overlayToggle?.classList.toggle("is-on", state.overlayVisible);
  e.invertToggle?.classList.toggle("is-on", state.invertSearch);
  if (e.combineSelect) {
    e.combineSelect.value = state.combineMethod;
    e.combineSelect.hidden = state.positivePoints.length < 2;
  }

  // Negative section
  if (e.negativeSection) e.negativeSection.hidden = state.negativePoints.length === 0;
  const negKey = state.negativePoints.map((p) => p.id).join(",");
  if (e.negativeList && state.negativePoints.length && negKey !== lastNegativeListKey) {
    lastNegativeListKey = negKey;
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
          lastNegativeListKey = ""; // force rebuild on next updateView
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
  const positiveKey = state.positivePoints.map((p) => p.id).join(",");
  if (positiveKey === lastPositiveListKey) {
    // data unchanged — skip rebuild to avoid flash
  } else {
  lastPositiveListKey = positiveKey;
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
          lastPositiveListKey = ""; // force rebuild on next updateView
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
  } // end positive list fingerprint block

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

  const resultKey = `${state.viewMode}|${visible.map((r) => r.chips_id).join(",")}`;
  if (resultKey !== lastResultListKey) {
  lastResultListKey = resultKey;
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
  } // end result list rebuild
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

function bboxToWKT(b: BBox): string {
  return `POLYGON((${b.west} ${b.south},${b.east} ${b.south},${b.east} ${b.north},${b.west} ${b.north},${b.west} ${b.south}))`;
}

function ringToWKT(ring: [number, number][]): string {
  return `POLYGON((${ring.map(([lng, lat]) => `${lng} ${lat}`).join(",")}))`;
}

function buildManifestQuery(bbox: BBox, polygon?: [number, number][]): string {
  const wkt = polygon ? ringToWKT(polygon) : bboxToWKT(bbox);
  return `
    SELECT path, rows, xmin, ymin, xmax, ymax, year
    FROM read_parquet(${sqlString(MANIFEST_URL)})
    WHERE ST_Intersects(
      ST_GeomFromText('${wkt}'),
      ST_MakeEnvelope(xmin, ymin, xmax, ymax)
    )
    ORDER BY rows DESC, path ASC
  `.trim();
}

function buildShardQuery(shardUrl: string, bbox: BBox, polygon?: [number, number][]): string {
  const wkt = polygon ? ringToWKT(polygon) : bboxToWKT(bbox);
  return `
    SELECT chips_id, bbox, embedding
    FROM read_parquet(${sqlString(shardUrl)})
    WHERE ST_Intersects(
      ST_GeomFromText('${wkt}'),
      ST_MakeEnvelope(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
    )
  `.trim();
}

async function fetchShardCandidates(
  db: duckdb.AsyncDuckDB,
  shardUrl: string,
  bbox: BBox,
  polygon?: [number, number][],
): Promise<CandidateRow[]> {
  const conn = await db.connect();
  try {
    const result = await conn.query(buildShardQuery(shardUrl, bbox, polygon));
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
    await conn.query("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;");
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

// Derive inverted results from cached base (non-inverted) scores without re-scoring.
// For no negatives: score_inv = L - score (hamming complement). With negatives: score_inv = -score.
// Both cases produce reversed sort order.
function applyInvert(base: RankedRow[]): RankedRow[] {
  if (!state.invertSearch || !base.length) return base;
  const hasNeg = state.negativePoints.length > 0;
  const L = base[0].embedding.length * 8;
  return [...base].map(r => ({ ...r, score: hasNeg ? -r.score : L - r.score })).reverse();
}

function resolveNegativeEmbeddings(): Uint8Array[] {
  return state.negativePoints.flatMap((point) => {
    if (point.embedding) return [point.embedding];
    const intersecting = state.candidateRows.filter((c) => containsPoint(c.bbox, point.lat, point.lng));
    if (!intersecting.length) return [];
    let best = intersecting[0];
    let bestD = Number.POSITIVE_INFINITY;
    for (const option of intersecting) {
      const c = centroid(option.bbox);
      const d = distanceSquared(point.lat, point.lng, c.lat, c.lng);
      if (d < bestD) { bestD = d; best = option; }
    }
    return [best.embedding];
  });
}

async function scoreWithWorker(exemplars: CandidateRow[]): Promise<RankedRow[]> {
  if (!scoringWorkerReady) return [];
  const worker = ensureScoringWorker();
  const requestId = ++scoringRequestId;
  const excludeIndices = new Set(exemplars.map((ex) => state.candidateRows.indexOf(ex)).filter((i) => i >= 0));
  let posEmbeddings = exemplars.map((ex) => new Uint8Array(ex.embedding));
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
  // in state.positivePoints and addRegion() will call scoreCandidates() again
  // once the last shard lands and the scoring worker is ready.
  if (state.loading || !scoringWorkerReady) {
    if (state.positivePoints.length) {
      if (state.loading) {
        setStatus(
          `Queued ${state.positivePoints.length} exemplar(s) — waiting for shards to finish downloading…`,
        );
      } else {
        // scoringWorkerReady=false but nothing downloading — no AOI loaded yet
        setStatus("Exemplar set — shift-drag to define a region and start searching.");
      }
    }
    return;
  }

  if (!state.candidateRows.length || !state.positivePoints.length) {
    state.positiveMatches = [];
    state.baseResults = [];
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
    state.baseResults = [];
    state.results = [];
    globe.setResults([], state.topK, state.viewMode);
    setStatus("No patch under the selected point — try closer to a tile center.");
    return;
  }

  setStatus(`Scoring ${new Intl.NumberFormat().format(state.candidateRows.length)} candidates…`);
  const scored = await scoreWithWorker(exemplars);
  if (runId !== latestScoreRunId) return;
  state.baseResults = scored;
  state.results = applyInvert(scored);
  if (state.overlayVisible) globe.setResults(scored, state.topK, state.viewMode);
  setStatus(`Ranked ${scored.length} candidates against ${exemplars.length} exemplar(s).`);
  updateView();
  // Background precompute gradient now that results exist
  void computeGradient(true);
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

async function computeOutliers(background = false): Promise<void> {
  if (!scoringWorkerReady || !state.candidateRows.length) return;
  if (state.outlierComputed || outlierComputing) return;
  outlierComputing = true;

  if (!background) setStatus("Computing outlier scores…");
  const worker = ensureScoringWorker();
  const requestId = ++scoringRequestId;
  try {
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
    if (state.viewMode === "outlier") {
      if (state.overlayVisible) globe.setResults(state.outlierResults, state.topK, "outlier");
      setStatus(`Outlier analysis: ${state.outlierResults.length} patches scored. Brightest = most unique.`);
      updateView();
    } else if (!background) {
      if (state.overlayVisible) globe.setResults(state.outlierResults, state.topK, "outlier");
      setStatus(`Outlier analysis: ${state.outlierResults.length} patches scored. Brightest = most unique.`);
      updateView();
    }
  } finally {
    outlierComputing = false;
  }
}

/* ---------------------------------------------------------- Surprise scoring */

async function computeSurprise(background = false): Promise<void> {
  if (!scoringWorkerReady || !state.candidateRows.length) return;
  if (state.surpriseComputed || surpriseComputing) return;
  surpriseComputing = true;

  if (!background) setStatus("Computing spatial surprise scores\u2026");
  const worker = ensureScoringWorker();
  const requestId = ++scoringRequestId;
  try {
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
    if (state.viewMode === "surprise") {
      if (state.overlayVisible) globe.setResults(state.surpriseResults, state.topK, "surprise");
      setStatus(`Surprise analysis: ${state.surpriseResults.length} patches scored. Brightest = most surprising.`);
      updateView();
    } else if (!background) {
      if (state.overlayVisible) globe.setResults(state.surpriseResults, state.topK, "surprise");
      setStatus(`Surprise analysis: ${state.surpriseResults.length} patches scored. Brightest = most surprising.`);
      updateView();
    }
  } finally {
    surpriseComputing = false;
  }
}

/* ---------------------------------------------------------- Gradient scoring */

async function computeGradient(background = false): Promise<void> {
  if (!scoringWorkerReady || !state.candidateRows.length || !state.results.length) return;
  if (gradientComputing) return;
  gradientComputing = true;

  if (!background) setStatus("Computing similarity gradient\u2026");
  const worker = ensureScoringWorker();
  const requestId = ++scoringRequestId;
  const scoreArr = new Float64Array(state.candidateRows.length);
  const scoreMap = new Map<string, number>();
  for (const r of state.results) scoreMap.set(r.chips_id, r.score);
  for (let i = 0; i < state.candidateRows.length; i++) {
    scoreArr[i] = scoreMap.get(state.candidateRows[i].chips_id) ?? 0;
  }
  try {
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
    if (state.viewMode === "gradient") {
      if (state.overlayVisible) globe.setResults(state.gradientResults, state.topK, "gradient");
      setStatus(`Gradient analysis: ${state.gradientResults.length} patches scored. Brightest = strongest boundary.`);
      updateView();
    } else if (!background) {
      if (state.overlayVisible) globe.setResults(state.gradientResults, state.topK, "gradient");
      setStatus(`Gradient analysis: ${state.gradientResults.length} patches scored. Brightest = strongest boundary.`);
      updateView();
    }
  } finally {
    gradientComputing = false;
  }
}

/* ---------------------------------------------------------- GeoParquet export */

// ── WKB point (little-endian, EPSG:4326 lon/lat as x/y) ─────────────────────
function wkbPointHex(lng: number, lat: number): string {
  const buf = new ArrayBuffer(21);
  const dv = new DataView(buf);
  dv.setUint8(0, 1); // byte order: little-endian
  dv.setUint32(1, 1, true); // geometry type: Point
  dv.setFloat64(5, lng, true);
  dv.setFloat64(13, lat, true);
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

// ── Minimal Thrift binary helpers to inject 'geo' into Parquet footer ────────

function concatBytes(parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

function thriftStrBytes(s: string): Uint8Array {
  const payload = new TextEncoder().encode(s);
  const len = new Uint8Array(4);
  new DataView(len.buffer).setInt32(0, payload.length, false); // BE
  return concatBytes([len, payload]);
}

// Encode a KeyValue struct: field 1 = key (STRING), field 2 = value (STRING), STOP
function thriftKVStruct(key: string, value: string): Uint8Array {
  return concatBytes([
    new Uint8Array([0x0b, 0x00, 0x01]),
    thriftStrBytes(key),
    new Uint8Array([0x0b, 0x00, 0x02]),
    thriftStrBytes(value),
    new Uint8Array([0x00]),
  ]);
}

// Encode field 5 (key_value_metadata) as a LIST<STRUCT> Thrift field
function thriftKVField(structs: Uint8Array[]): Uint8Array {
  const header = new Uint8Array(8);
  header[0] = 0x0f; // type = LIST
  header[1] = 0x00;
  header[2] = 0x05; // field id = 5
  header[3] = 0x0c; // elem type = STRUCT
  new DataView(header.buffer).setInt32(4, structs.length, false); // BE count
  return concatBytes([header, ...structs]);
}

// Walk Thrift binary to find an existing key_value_metadata LIST field (field 5).
// Returns { countPos, dataEnd, count } if found, else null.
function findThriftKVMeta(footer: Uint8Array): { countPos: number; dataEnd: number; count: number } | null {
  const dv = new DataView(footer.buffer, footer.byteOffset, footer.byteLength);
  let pos = 0;

  function skipValue(type: number): void {
    if (type === 2 || type === 3) {
      pos += 1;
    } else if (type === 4) {
      pos += 8;
    } else if (type === 6) {
      pos += 2;
    } else if (type === 8) {
      pos += 4;
    } else if (type === 10) {
      pos += 8;
    } else if (type === 11) {
      pos += 4 + dv.getInt32(pos, false);
    } else if (type === 12) {
      let t: number;
      while ((t = footer[pos++]) !== 0) {
        pos += 2; // field id
        skipValue(t);
      }
    } else if (type === 13) {
      const kt = footer[pos++];
      const vt = footer[pos++];
      const n = dv.getInt32(pos, false);
      pos += 4;
      for (let i = 0; i < n; i++) {
        skipValue(kt);
        skipValue(vt);
      }
    } else if (type === 14 || type === 15) {
      const et = footer[pos++];
      const n = dv.getInt32(pos, false);
      pos += 4;
      for (let i = 0; i < n; i++) skipValue(et);
    }
  }

  while (pos < footer.length) {
    const type = footer[pos++];
    if (type === 0) return null; // STOP — field 5 not present
    const fieldId = dv.getInt16(pos, false); // BE
    pos += 2;
    if (fieldId === 5 && type === 0x0f) {
      // LIST — read element type then count
      pos += 1; // elem type (0x0c = STRUCT)
      const countPos = pos;
      const count = dv.getInt32(pos, false);
      pos += 4;
      // skip all existing structs
      for (let i = 0; i < count; i++) skipValue(0x0c);
      return { countPos, dataEnd: pos, count };
    }
    skipValue(type);
  }
  return null;
}

// Inject (or append to) the 'geo' key in a Parquet footer's Thrift binary.
function injectGeoMeta(footer: Uint8Array, geoJson: string): Uint8Array {
  const geo = thriftKVStruct("geo", geoJson);
  const meta = findThriftKVMeta(footer);

  if (meta) {
    // field 5 already exists — increment count and append our struct
    const { countPos, dataEnd, count } = meta;
    const newCount = new Uint8Array(4);
    new DataView(newCount.buffer).setInt32(0, count + 1, false);
    return concatBytes([
      footer.slice(0, countPos),
      newCount,
      footer.slice(countPos + 4, dataEnd),
      geo,
      footer.slice(dataEnd),
    ]);
  } else {
    // field 5 absent — insert before the FileMetaData STOP byte (last byte of footer)
    return concatBytes([footer.slice(0, footer.length - 1), thriftKVField([geo]), new Uint8Array([0x00])]);
  }
}

// Patch a DuckDB-written Parquet buffer to add valid GeoParquet metadata.
function makeGeoParquet(buf: Uint8Array, geoJson: string): Uint8Array {
  const n = buf.length;
  const dv = new DataView(buf.buffer, buf.byteOffset);
  const footerLen = dv.getInt32(n - 8, true); // LE
  const footerStart = n - 8 - footerLen;

  const prefix = buf.slice(0, footerStart);
  const footer = buf.slice(footerStart, footerStart + footerLen);
  const newFooter = injectGeoMeta(footer, geoJson);

  const newLen = new Uint8Array(4);
  new DataView(newLen.buffer).setInt32(0, newFooter.length, true); // LE
  return concatBytes([prefix, newFooter, newLen, new Uint8Array([0x50, 0x41, 0x52, 0x31])]); // PAR1
}

const GEO_META = JSON.stringify({
  version: "1.0.0",
  primary_column: "geometry",
  columns: { geometry: { encoding: "WKB", geometry_types: ["Point"] } },
});

async function exportGeoParquet(): Promise<void> {
  const exemplars = state.positiveMatches.map((m) => m.candidate);
  const topk = state.results.slice(0, state.viewMode === "topk" ? state.topK : state.results.length);
  if (!exemplars.length && !topk.length) return;

  setStatus("Exporting GeoParquet…");
  try {
    const db = await getDuckDB();
    const conn = await db.connect();

    // Build VALUES clauses; geometry = WKB point (lng, lat) as hex literal
    const rows: string[] = [];
    for (const ex of exemplars) {
      const c = centroid(ex.bbox);
      rows.push(
        `('exemplar', ${sqlString(ex.chips_id)}, ${c.lat}, ${c.lng}, ${ex.bbox.west}, ${ex.bbox.south}, ${ex.bbox.east}, ${ex.bbox.north}, NULL::DOUBLE, NULL::INT, x'${wkbPointHex(c.lng, c.lat)}'::BLOB)`,
      );
    }
    for (const [i, r] of topk.entries()) {
      const c = centroid(r.bbox);
      rows.push(
        `('candidate', ${sqlString(r.chips_id)}, ${c.lat}, ${c.lng}, ${r.bbox.west}, ${r.bbox.south}, ${r.bbox.east}, ${r.bbox.north}, ${r.score}, ${i + 1}, x'${wkbPointHex(c.lng, c.lat)}'::BLOB)`,
      );
    }

    const vfsPath = "/tmp/export.parquet";
    await conn.query(`
      COPY (
        SELECT * FROM (
          VALUES ${rows.join(",\n")}
        ) AS t(type, chips_id, lat, lng, west, south, east, north, score, rank, geometry)
      ) TO '${vfsPath}' (FORMAT PARQUET, COMPRESSION ZSTD)
    `);

    const raw = await db.copyFileToBuffer(vfsPath);
    await conn.close();

    const buf = makeGeoParquet(raw, GEO_META);

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

async function addRegion(bbox: BBox, polygon?: [number, number][]): Promise<void> {
  const id = state.nextAoiId++;
  const runId = 1;
  regionLoadRunIds.set(id, runId);

  state.bboxes.push({ id, bbox, ...(polygon ? { polygon } : {}) });
  state.regionRows.set(id, []);
  state.regionShardCounts.set(id, 0);
  state.loading = true;
  // Clear derived results so stale overlay doesn't linger
  resetComputeState();
  state.baseResults = [];
  state.results = [];
  state.outlierResults = [];
  state.outlierComputed = false;
  state.surpriseResults = [];
  state.surpriseComputed = false;
  state.gradientResults = [];
  state.threshold = Infinity;
  state.positiveMatches = [];
  globe.setAois(state.bboxes);
  globe.setPositiveMatches([]);
  globe.setResults([], state.topK, state.viewMode);
  globe.setPreview(null);
  globe.fitBounds(bbox, { padding: 60 });
  setStatus("Fetching intersecting shards\u2026");
  updateView();

  try {
    const db = await getDuckDB();
    const conn = await db.connect();
    const manifestResult = await conn.query(buildManifestQuery(bbox, polygon));
    const shards = manifestResult.toArray() as ManifestRow[];
    await conn.close();
    if (regionLoadRunIds.get(id) !== runId) { state.loading = false; return; }

    state.regionShardCounts.set(id, shards.length);
    updateView();
    if (!shards.length) {
      state.loading = false;
      setStatus("No shards intersect that region.");
      return;
    }

    setStatus(`Loading patches from ${shards.length} shard(s) for AOI ${id}…`);

    // Query each shard in parallel with bounded concurrency.
    let completed = 0;
    const shardUrls = shards.map((s) => resolveShardUrl(s.path));
    const regionAll: CandidateRow[] = [];
    const settled = await mapWithConcurrency(shardUrls, 8, async (url) => {
      const rows = await fetchShardCandidates(db, url, bbox, polygon);
      if (regionLoadRunIds.get(id) !== runId) return rows;
      regionAll.push(...rows);
      state.regionRows.get(id)!.push(...rows);
      state.candidateRows = [...state.regionRows.values()].flat();
      completed += 1;
      setStatus(`AOI ${id} — ${completed}/${shards.length} shards · ${new Intl.NumberFormat().format(state.candidateRows.length)} total patches`);
      return rows;
    });
    if (regionLoadRunIds.get(id) !== runId) { state.loading = false; return; }

    const failed = settled.filter((r) => r.status === "rejected");
    if (failed.length) console.warn(`AOI ${id} shard fetch failures:`, failed);

    state.regionRows.set(id, regionAll);
    state.candidateRows = [...state.regionRows.values()].flat();
    initScoringWorker(state.candidateRows);
    state.loading = false;
    const totalPatches = new Intl.NumberFormat().format(state.candidateRows.length);
    const base = regionAll.length
      ? `AOI ${id} loaded — ${totalPatches} total patches. Click anywhere to seed an exemplar.`
      : `AOI ${id} loaded, but no patches returned.`;
    setStatus(failed.length ? `${base} (${failed.length} shard(s) failed)` : base);
    updateView();

    if (state.positivePoints.length && state.candidateRows.length) {
      void scoreCandidates();
    }
    // Background precompute for region-only views (no exemplars needed)
    void computeOutliers(true);
    void computeSurprise(true);
  } catch (err) {
    state.loading = false;
    if (regionLoadRunIds.get(id) !== runId) return;
    setStatus(`Failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

function removeRegion(id: number): void {
  lastPositiveListKey = "";
  lastNegativeListKey = "";
  lastResultListKey = "";
  state.bboxes = state.bboxes.filter((e) => e.id !== id);
  state.regionRows.delete(id);
  state.regionShardCounts.delete(id);
  regionLoadRunIds.delete(id);
  // If no more regions are loading, clear the flag so scoreCandidates() isn't stuck.
  if (!state.bboxes.length) state.loading = false;
  resetComputeState();
  state.candidateRows = [...state.regionRows.values()].flat();
  state.baseResults = [];
  state.results = [];
  state.outlierResults = [];
  state.outlierComputed = false;
  state.surpriseResults = [];
  state.surpriseComputed = false;
  state.gradientResults = [];
  state.positiveMatches = [];
  globe.setAois(state.bboxes);
  globe.setPositiveMatches([]);
  globe.setResults([], state.topK, state.viewMode);
  if (state.candidateRows.length) {
    initScoringWorker(state.candidateRows);
    if (state.positivePoints.length) void scoreCandidates();
  }
  setStatus(state.bboxes.length ? "Region removed. Remaining regions active." : "Cleared. Shift-drag to define a new region.");
  updateView();
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

    // Phase 1: fetch chips_id + bbox only (no embedding) to find the target row
    for (const shard of shards) {
      const url = resolveShardUrl(shard.path);
      const p1 = await conn.query(`
        SELECT chips_id, bbox
        FROM read_parquet(${sqlString(url)})
        WHERE bbox.xmax >= ${lng - 0.01} AND bbox.xmin <= ${lng + 0.01}
          AND bbox.ymax >= ${lat - 0.01} AND bbox.ymin <= ${lat + 0.01}
      `.trim());
      const candidates = p1.toArray() as Array<{
        chips_id: string;
        bbox: { xmin: number; ymin: number; xmax: number; ymax: number };
      }>;
      const hit = candidates.find((r) => containsPoint(normalizeBBox(r.bbox), lat, lng));
      if (!hit) continue;

      // Phase 2: fetch just the embedding for that one chips_id
      const p2 = await conn.query(`
        SELECT embedding
        FROM read_parquet(${sqlString(url)})
        WHERE chips_id = ${sqlString(hit.chips_id)}
        LIMIT 1
      `.trim());
      const embRow = p2.toArray()[0] as { embedding: unknown } | undefined;
      if (!embRow) continue;

      return {
        chips_id: hit.chips_id,
        bbox: normalizeBBox(hit.bbox),
        embedding: normalizeEmbedding(embRow.embedding),
        shard_path: url,
      };
    }
    return null;
  } finally {
    await conn.close();
  }
}

function addPositive(lat: number, lng: number): void {
  const insideBbox = isInsideAnyAoi(lat, lng);
  const candidatesReady = state.candidateRows.length > 0;

  if (insideBbox && candidatesReady) {
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
  } else if (insideBbox) {
    // Region is defined but still loading — queue point so it's ready when shards land
    state.positivePoints.push({ id: state.positivePoints.length + 1, lat, lng });
    globe.setPositives(state.positivePoints);
    void scoreCandidates(); // returns early + shows "queued N exemplar(s)" status
    updateView();
  } else {
    // External exemplar — show dot immediately, fetch embedding in background
    const placeholderId = state.positivePoints.length + 1;
    state.positivePoints.push({ id: placeholderId, lat, lng });
    globe.setPositives(state.positivePoints);
    setStatus("Fetching external exemplar embedding…");
    updateView();
    void fetchExternalEmbedding(lat, lng).then((row) => {
      if (!row) {
        // Remove placeholder
        state.positivePoints = state.positivePoints.filter((p) => p.id !== placeholderId);
        globe.setPositives(state.positivePoints);
        setStatus("No patch found at that location.");
        updateView();
        return;
      }
      // Deduplicate by chips_id
      const existing = state.positivePoints.some((p) => p.embedding && p.chips_id === row.chips_id);
      if (existing) {
        state.positivePoints = state.positivePoints.filter((p) => p.id !== placeholderId);
        globe.setPositives(state.positivePoints);
        setStatus("That patch is already selected.");
        updateView();
        return;
      }
      // Update placeholder in-place with embedding
      const placeholder = state.positivePoints.find((p) => p.id === placeholderId);
      if (placeholder) {
        placeholder.embedding = row.embedding;
        placeholder.chips_id = row.chips_id;
      }
      globe.setPositives(state.positivePoints);
      void scoreCandidates();
      updateView();
    });
  }
}

function addNegative(lat: number, lng: number): void {
  const insideBbox = isInsideAnyAoi(lat, lng);
  const candidatesReady = state.candidateRows.length > 0;

  if (insideBbox && candidatesReady) {
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
  } else if (insideBbox) {
    // Still loading — queue and draw immediately
    state.negativePoints.push({ id: state.negativePoints.length + 1, lat, lng });
    globe.setNegatives(state.negativePoints);
    void scoreCandidates();
    updateView();
  } else {
    // Outside AOI — show dot immediately, fetch embedding in background
    const placeholderId = state.negativePoints.length + 1;
    state.negativePoints.push({ id: placeholderId, lat, lng });
    globe.setNegatives(state.negativePoints);
    setStatus("Fetching external negative embedding…");
    updateView();
    void fetchExternalEmbedding(lat, lng).then((row) => {
      if (!row) {
        state.negativePoints = state.negativePoints.filter((p) => p.id !== placeholderId);
        globe.setNegatives(state.negativePoints);
        setStatus("No patch found at that location.");
        updateView();
        return;
      }
      const existing = state.negativePoints.some((p) => p.embedding && p.chips_id === row.chips_id);
      if (existing) {
        state.negativePoints = state.negativePoints.filter((p) => p.id !== placeholderId);
        globe.setNegatives(state.negativePoints);
        setStatus("That patch is already a negative.");
        updateView();
        return;
      }
      const placeholder = state.negativePoints.find((p) => p.id === placeholderId);
      if (placeholder) {
        placeholder.embedding = row.embedding;
        placeholder.chips_id = row.chips_id;
      }
      globe.setNegatives(state.negativePoints);
      void scoreCandidates();
      updateView();
    });
  }
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
  lastPositiveListKey = "";
  lastNegativeListKey = "";
  lastResultListKey = "";
  gradientComputing = false;
  state.positiveMatches = [];
  state.baseResults = [];
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

function clearAllRegions(): void {
  lastPositiveListKey = "";
  lastNegativeListKey = "";
  lastResultListKey = "";
  state.bboxes = [];
  state.nextAoiId = 1;
  state.regionRows = new Map();
  state.regionShardCounts = new Map();
  regionLoadRunIds.clear();
  state.loading = false;
  state.candidateRows = [];
  state.positivePoints = [];
  state.negativePoints = [];
  state.positiveMatches = [];
  resetComputeState();
  state.baseResults = [];
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
  globe.setAois([]);
  globe.setPositives([]);
  globe.setNegatives([]);
  globe.setPositiveMatches([]);
  globe.setResults([], state.topK, state.viewMode);
  globe.setPreview(null);
  setStatus("Cleared. Shift-drag to define a new region.");
  updateView();
}

/* --------------------------------------------------------------- Bootstrap */

function wire(): void {
  const e = els();
  e.drawBtn?.addEventListener("click", () => {
    const wasArmed = globe.isArmed() && globe.getDrawMode() === "rect";
    globe.armDraw(!wasArmed, "rect");
    setStatus(!wasArmed ? "Draw armed — drag on the globe to define a region." : "Draw disarmed.");
    updateView();
  });
  e.drawPolyBtn?.addEventListener("click", () => {
    const wasArmed = globe.isArmed() && globe.getDrawMode() === "polygon";
    globe.armDraw(!wasArmed, "polygon");
    setStatus(!wasArmed ? "Polygon draw armed — click to add vertices, double-click to close." : "Draw disarmed.");
    updateView();
  });
  e.zoomRegionBtn?.addEventListener("click", () => {
    if (!state.bboxes.length) return;
    const union: BBox = {
      west: Math.min(...state.bboxes.map((e) => e.bbox.west)),
      south: Math.min(...state.bboxes.map((e) => e.bbox.south)),
      east: Math.max(...state.bboxes.map((e) => e.bbox.east)),
      north: Math.max(...state.bboxes.map((e) => e.bbox.north)),
    };
    globe.fitBounds(union, { padding: 60 });
  });
  e.clearPointsBtn?.addEventListener("click", clearPoints);
  e.exportBtn?.addEventListener("click", () => void exportGeoParquet());
  e.overlayToggle?.addEventListener("click", () => {
    state.overlayVisible = !state.overlayVisible;
    applyOverlay();
    e.overlayToggle?.classList.toggle("is-on", state.overlayVisible);
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
      if (mode === "outlier" && state.outlierComputed) {
        globe.setResults(state.outlierResults, state.topK, mode);
        setStatus(`Outlier view — ${state.outlierResults.length} patches scored. Brightest = most unique.`);
      } else if (mode === "outlier" && outlierComputing) {
        setStatus("Computing outlier scores… results will appear shortly.");
      } else if (mode === "outlier") {
        void computeOutliers();
      } else if (mode === "surprise" && state.surpriseComputed) {
        globe.setResults(state.surpriseResults, state.topK, mode);
        setStatus(`Surprise view — ${state.surpriseResults.length} patches scored. Brightest = most spatially anomalous.`);
      } else if (mode === "surprise" && surpriseComputing) {
        setStatus("Computing surprise scores… results will appear shortly.");
      } else if (mode === "surprise") {
        void computeSurprise();
      } else if (mode === "gradient") {
        if (gradientComputing) {
          setStatus("Computing edge scores… results will appear shortly.");
        } else if (state.gradientResults.length) {
          globe.setResults(state.gradientResults, state.topK, mode);
          setStatus(`Edge view — ${state.gradientResults.length} patches scored. Brightest = strongest boundary.`);
        } else if (state.results.length) {
          void computeGradient();
        } else {
          setStatus("Edge view — add exemplars first to compute similarity gradients.");
        }
      } else if (mode === "threshold") {
        const filtered = state.results.filter((r) => r.score <= state.threshold);
        globe.setResults(filtered, filtered.length, mode);
        setStatus(`Cutoff view — adjust the threshold slider to filter patches by distance.`);
      } else {
        globe.setResults(state.results, state.topK, mode);
        if (state.results.length) setStatus(`Top-K view — showing ${Math.min(state.topK, state.results.length)} of ${state.results.length} ranked patches.`);
      }
      updateView();
    });
  });

  // Invert toggle — derive from cached base results, no re-scoring needed
  e.invertToggle?.addEventListener("click", () => {
    state.invertSearch = !state.invertSearch;
    if (state.baseResults.length) {
      state.results = applyInvert(state.baseResults);
      state.threshold = Infinity;
      if (state.overlayVisible) globe.setResults(state.results, state.topK, state.viewMode);
    } else {
      void scoreCandidates();
    }
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
    if (state.bboxes.length) clearAllRegions();
  });
}

function bootstrap(): void {
  renderShell();
  const mapEl = document.querySelector<HTMLDivElement>("#map");
  if (!mapEl) throw new Error("#map missing");
  globe = new GlobeMap(mapEl, {
    onDrawComplete: ({ bbox, polygon }) => {
      void addRegion(bbox, polygon);
    },
    onAoiClick: (lat, lng) => addPositive(lat, lng),
    onNegativeClick: (lat, lng) => addNegative(lat, lng),
    onResultHover: (row) => globe.setPreview(row),
    onResultPick: (row) => {
      const c = centroid(row.bbox);
      addPositive(c.lat, c.lng);
    },
    getBBox: () => state.bboxes[state.bboxes.length - 1]?.bbox ?? null,
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
  onEnter?: () => void;   // side-effect fired when step becomes active
};

/* ---- Tutorial simulation helpers ---- */

// Kansas AOI used throughout the demo
const DEMO_BBOX: BBox = { west: -98.6, east: -98.1, south: 38.45, north: 38.95 };
// Positive: large open field near center
const DEMO_POS_LAT = 38.72;
const DEMO_POS_LNG = -98.34;
// Negative: different land type (sandy/barren patch to the southwest)
const DEMO_NEG_LAT = 38.52;
const DEMO_NEG_LNG = -98.54;

let _tutSimHandles: ReturnType<typeof setTimeout>[] = [];

function tutSimDelay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    const h = setTimeout(resolve, ms);
    _tutSimHandles.push(h);
  });
}

function tutCancelSim(): void {
  _tutSimHandles.forEach(clearTimeout);
  _tutSimHandles = [];
}

async function tutSimulateDraw(bbox: BBox, durationMs = 1400): Promise<void> {
  const steps = 32;
  const interval = durationMs / steps;
  for (let i = 1; i <= steps; i++) {
    if (!tutorialState.active) return;
    const t = i / steps;
    globe.setDraft({
      west: bbox.west,
      east: bbox.west + (bbox.east - bbox.west) * t,
      south: bbox.north - (bbox.north - bbox.south) * t,
      north: bbox.north,
    });
    await tutSimDelay(interval);
  }
  globe.setDraft(null);
  await addRegion(bbox);
}

function tutShowRipple(lat: number, lng: number, color: string): void {
  const pt = globe.map.project([lng, lat]);
  const container = globe.map.getCanvasContainer();
  const cRect = container.getBoundingClientRect();
  const ripple = document.createElement("div");
  ripple.className = "tut-ripple";
  ripple.style.left = `${cRect.left + pt.x}px`;
  ripple.style.top = `${cRect.top + pt.y}px`;
  ripple.style.setProperty("--tut-ripple-color", color);
  document.body.appendChild(ripple);
  setTimeout(() => ripple.remove(), 900);
}

async function tutSimulatePositive(lat: number, lng: number): Promise<void> {
  tutShowRipple(lat, lng, "#c74633");
  await tutSimDelay(180);
  addPositive(lat, lng);
}

// Place an exemplar just outside the AOI using a loaded candidate's embedding —
// guaranteed to resolve (no HTTP fetch), shown slightly north of the bbox edge.
async function tutSimulateExternalPositive(): Promise<void> {
  const candidates = state.candidateRows;
  if (!candidates.length) return;
  // Pick a candidate near the north edge of the demo bbox
  const sorted = candidates.slice().sort((a, b) => {
    const latA = (a.bbox.north + a.bbox.south) / 2;
    const latB = (b.bbox.north + b.bbox.south) / 2;
    return Math.abs(DEMO_BBOX.north - latA) - Math.abs(DEMO_BBOX.north - latB);
  });
  const pick = sorted[0];
  const c = centroid(pick.bbox);
  // Visual marker sits just above the bbox north edge — clearly outside the region
  const markerLat = DEMO_BBOX.north + 0.08;
  const markerLng = c.lng;
  tutShowRipple(markerLat, markerLng, "#c74633");
  await tutSimDelay(180);
  const id = state.positivePoints.length + 1;
  state.positivePoints.push({ id, lat: markerLat, lng: markerLng, embedding: pick.embedding, chips_id: pick.chips_id });
  globe.setPositives(state.positivePoints);
  void scoreCandidates();
  updateView();
}

async function tutSimulateNegative(lat: number, lng: number): Promise<void> {
  tutShowRipple(lat, lng, "#3b82f6");
  await tutSimDelay(180);
  addNegative(lat, lng);
}

function tutWaitForData(maxWaitMs = 35000): Promise<boolean> {
  return new Promise((resolve) => {
    const deadline = Date.now() + maxWaitMs;
    const poll = () => {
      if (state.candidateRows.length > 0) { resolve(true); return; }
      if (Date.now() > deadline) { resolve(false); return; }
      const h = setTimeout(poll, 300);
      _tutSimHandles.push(h);
    };
    poll();
  });
}

function tutSetViewMode(mode: ViewMode): void {
  const tab = document.querySelector<HTMLButtonElement>(`.view-tab[data-view="${mode}"]`);
  tab?.click();
}

const TUTORIAL_STEPS: TutorialStep[] = [
  {
    title: "Welcome to terrabit",
    body: "terrabit finds every satellite patch on Earth that looks like a location you point at — powered by compact binary embeddings. Watch this live demo to see how it works.",
    placement: "center",
  },
  {
    title: "Step 1 — fly to a region",
    body: "We're zooming to the agricultural plains of central Kansas — a compact, high-contrast area that loads in seconds. Watch as terrabit draws a region and loads the embeddings.",
    placement: "center",
    onEnter: () => {
      globe.map.flyTo({ center: [-98.35, 38.7], zoom: 9, duration: 1600 });
      setTimeout(() => { void tutSimulateDraw(DEMO_BBOX); }, 1800);
    },
  },
  {
    target: "#positive-list",
    title: "Step 2 — place a positive exemplar",
    body: "Patches are loaded. We click a farm field <em>inside</em> the region, then a second point just <em>outside</em> it — <strong>exemplars can go anywhere on the globe</strong>. terrabit fetches the external embedding on the fly and scores every patch by binary Hamming distance.",
    placement: "right",
    padding: 8,
    onEnter: () => {
      // Don't re-run if navigating back — exemplars already placed
      if (state.positivePoints.length > 0) return;
      const stepAtEnter = tutorialState.step;
      void tutWaitForData().then(async (ready) => {
        if (!ready || !tutorialState.active || tutorialState.step !== stepAtEnter) return;
        await tutSimulatePositive(DEMO_POS_LAT, DEMO_POS_LNG);
        await tutSimDelay(2200);
        if (!tutorialState.active || tutorialState.step !== stepAtEnter) return;
        // Second exemplar just north of the drawn bbox — visible on screen, injected directly
        await tutSimulateExternalPositive();
      });
    },
  },
  {
    target: "#negative-section",
    title: "Step 3 — add a negative to refine",
    body: "We're right-clicking a different land type as a negative exemplar. Negatives push <em>away</em> from that pattern — results immediately shift to emphasize the positive and suppress the negative.",
    placement: "right",
    padding: 8,
    onEnter: () => {
      if (state.negativePoints.length > 0) return;
      void tutSimulateNegative(DEMO_NEG_LAT, DEMO_NEG_LNG);
    },
  },
  {
    target: ".view-toggle",
    title: "Step 4 — explore view modes",
    body: "Watch as terrabit cycles through every view: <strong>Top-K</strong> ranks closest matches · <strong>Heat</strong> paints a continuous similarity surface · <strong>Cutoff</strong> filters by distance · <strong>Outlier</strong> finds unique patches · <strong>Surprise</strong> spots spatial anomalies · <strong>Edge</strong> traces similarity boundaries.",
    placement: "right",
    padding: 10,
    onEnter: () => {
      const stepAtEnter = tutorialState.step;
      const modes: ViewMode[] = ["topk", "heatmap", "threshold", "outlier", "surprise", "gradient"];
      const cycle = async (): Promise<void> => {
        for (const mode of modes) {
          if (!tutorialState.active || tutorialState.step !== stepAtEnter) return;
          tutSetViewMode(mode);
          await tutSimDelay(1100);
        }
        if (!tutorialState.active || tutorialState.step !== stepAtEnter) return;
        void cycle();
      };
      void tutSimDelay(600).then(() => {
        if (!tutorialState.active || tutorialState.step !== stepAtEnter) return;
        void cycle();
      });
    },
  },
  {
    title: "You're ready",
    body: "Click anywhere on the globe to add your own exemplars. Add more regions, flip <strong>Invert</strong> to find opposites, or explore <strong>Outlier</strong>, <strong>Surprise</strong>, and <strong>Edge</strong> views. Export results as GeoParquet for QGIS, DuckDB, or GeoPandas.",
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
  step.onEnter?.();

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
  tutCancelSim();
  clearAllRegions();
  tutorialState.active = true;
  tutorialState.step = 0;
  tutorialRender();
}

function tutorialStop(): void {
  tutCancelSim();
  tutorialState.active = false;
  const overlay = document.querySelector<HTMLElement>("#tut-overlay");
  const card = document.querySelector<HTMLElement>("#tut-card");
  if (overlay) overlay.classList.remove("is-active", "has-spotlight", "is-transitioning");
  if (card) card.classList.remove("is-active", "is-transitioning");
  // If data never finished loading (early skip), clear the half-baked state
  if (state.candidateRows.length === 0) clearAllRegions();
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
