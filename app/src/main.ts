import * as duckdb from "@duckdb/duckdb-wasm";
import duckdbWasmEh from "@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url";
import duckdbWasmMvp from "@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url";
import duckdbWorkerEh from "@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url";
import duckdbWorkerMvp from "@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url";
import * as L from "leaflet";
import "leaflet/dist/leaflet.css";

import "./styles.css";

const MANIFEST_URL =
  "https://data.source.coop/geovibes/terrabit/clay-v1_5-binary-sentinel-2/manifest.parquet";
const DEFAULT_TOP_K = 10;
const MAX_TOP_K = 50;
const MAX_MANIFEST_SHARDS = 256;
const MAX_AOI_ROWS = 50000;
const SENTINEL_2024_TILES =
  "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg";
const OSM_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const SENTINEL_ATTRIBUTION =
  'Sentinel-2 cloudless - <a href="https://s2maps.eu" target="_blank" rel="noreferrer">https://s2maps.eu</a> by <a href="https://eox.at" target="_blank" rel="noreferrer">EOX IT Services GmbH</a> (Contains modified Copernicus Sentinel data 2024)';
const OSM_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors';

type BBox = {
  west: number;
  south: number;
  east: number;
  north: number;
};

type ManifestRow = {
  path: string;
  rows: number;
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  year?: string;
};

type PositivePoint = {
  id: number;
  lat: number;
  lng: number;
};

type PositiveMatch = {
  pointId: number;
  candidate: CandidateRow;
};

type CandidateRow = {
  chips_id: string;
  bbox: BBox;
  embedding: Uint8Array;
  shard_path: string;
};

type RankedRow = CandidateRow & {
  score: number;
};

type AppState = {
  manifestUrl: string;
  bbox: BBox | null;
  status: string;
  controlsCollapsed: boolean;
  manifestShards: ManifestRow[];
  candidateRows: CandidateRow[];
  positivePoints: PositivePoint[];
  positiveMatches: PositiveMatch[];
  results: RankedRow[];
  shardCount: number;
  candidateCount: number;
  topK: number;
  showHeatmap: boolean;
  isBusy: boolean;
};

type WorkerScoreResult = {
  index: number;
  score: number;
};

const state: AppState = {
  manifestUrl: getDefaultManifestUrl(),
  bbox: null,
  status: "Draw a region with the button or Shift-drag. Then click inside it to add positive points.",
  controlsCollapsed: false,
  manifestShards: [],
  candidateRows: [],
  positivePoints: [],
  positiveMatches: [],
  results: [],
  shardCount: 0,
  candidateCount: 0,
  topK: DEFAULT_TOP_K,
  showHeatmap: false,
  isBusy: false,
};

let dbPromise: Promise<duckdb.AsyncDuckDB> | null = null;
let popcountTable: Uint8Array | null = null;
let positiveLayer: any = null;
let positiveMatchLayer: any = null;
let aoiLayer: any = null;
let resultLayer: any = null;
let heatmapLayer: any = null;
let heatmapRenderer: any = null;
let previewLayer: any = null;
let mapRef: any = null;
let drawStartLatLng: { lat: number; lng: number } | null = null;
let draftRectangle: any = null;
let drawMoved = false;
let drawModeArmed = false;
let suppressNextMapClick = false;
let scoringWorker: Worker | null = null;
let scoringWorkerReady = false;
let scoringRequestId = 0;
let latestScoreRunId = 0;

const DUCKDB_BUNDLES: duckdb.DuckDBBundles = {
  mvp: {
    mainModule: duckdbWasmMvp,
    mainWorker: duckdbWorkerMvp,
  },
  eh: {
    mainModule: duckdbWasmEh,
    mainWorker: duckdbWorkerEh,
  },
};

function getDefaultManifestUrl(): string {
  return MANIFEST_URL;
}

function sqlString(value: string): string {
  return `'${value.replaceAll("'", "''")}'`;
}

function normalizeBBox(value: unknown): BBox {
  const box = value as { xmin: number; ymin: number; xmax: number; ymax: number };
  return {
    west: box.xmin,
    south: box.ymin,
    east: box.xmax,
    north: box.ymax,
  };
}

function normalizeEmbedding(value: unknown): Uint8Array {
  if (value instanceof Uint8Array) {
    return value;
  }
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength));
  }
  if (Array.isArray(value)) {
    return Uint8Array.from(value.map((item) => Number(item)));
  }
  if (typeof value === "object" && value !== null) {
    const candidate = value as {
      toArray?: () => unknown;
      values?: () => Iterable<unknown>;
      length?: number;
      [index: number]: unknown;
    };
    if (typeof candidate.toArray === "function") {
      return normalizeEmbedding(candidate.toArray());
    }
    if (typeof candidate.values === "function") {
      return normalizeEmbedding(Array.from(candidate.values()));
    }
    if (typeof candidate.length === "number") {
      return Uint8Array.from(Array.from({ length: candidate.length }, (_, index) => Number(candidate[index])));
    }
  }
  throw new TypeError("Unexpected embedding value");
}

function overlaps(a: BBox, b: BBox): boolean {
  return a.east >= b.west && a.west <= b.east && a.north >= b.south && a.south <= b.north;
}

function containsPoint(box: BBox, lat: number, lng: number): boolean {
  return lng >= box.west && lng <= box.east && lat >= box.south && lat <= box.north;
}

function centroid(box: BBox): { lat: number; lng: number } {
  return {
    lat: (box.south + box.north) / 2,
    lng: (box.west + box.east) / 2,
  };
}

function distanceSquared(aLat: number, aLng: number, bLat: number, bLng: number): number {
  const dLat = aLat - bLat;
  const dLng = aLng - bLng;
  return dLat * dLat + dLng * dLng;
}

function resolvePositiveMatches(): PositiveMatch[] {
  return state.positivePoints.flatMap((point) => {
    const intersecting = state.candidateRows.filter((candidate) => containsPoint(candidate.bbox, point.lat, point.lng));
    if (!intersecting.length) {
      return [];
    }
    let candidate = intersecting[0];
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const option of intersecting) {
      const center = centroid(option.bbox);
      const distance = distanceSquared(point.lat, point.lng, center.lat, center.lng);
      if (distance < bestDistance) {
        bestDistance = distance;
        candidate = option;
      }
    }
    return [{ pointId: point.id, candidate }];
  });
}

function buildManifestQuery(manifestUrl: string, bbox: BBox): string {
  return `
    SELECT path, rows, xmin, ymin, xmax, ymax, year
    FROM read_parquet(${sqlString(manifestUrl)})
    WHERE xmax >= ${bbox.west}
      AND xmin <= ${bbox.east}
      AND ymax >= ${bbox.south}
      AND ymin <= ${bbox.north}
    ORDER BY rows DESC, path ASC
    LIMIT ${MAX_MANIFEST_SHARDS}
  `.trim();
}

function buildCandidateQuery(manifestUrl: string, shards: ManifestRow[], bbox: BBox): string {
  const shardUrls = shards.map((shard) => sqlString(resolveShardUrl(manifestUrl, shard.path)));
  return `
    SELECT chips_id, bbox, embedding, filename AS shard_path
    FROM read_parquet([${shardUrls.join(", ")}], filename=true)
    WHERE bbox.xmax >= ${bbox.west}
      AND bbox.xmin <= ${bbox.east}
      AND bbox.ymax >= ${bbox.south}
      AND bbox.ymin <= ${bbox.north}
    LIMIT ${MAX_AOI_ROWS}
  `.trim();
}

function resolveShardUrl(manifestUrl: string, relativePath: string): string {
  try {
    return new URL(relativePath, manifestUrl).toString();
  } catch {
    return relativePath;
  }
}

function ensurePopcount(): Uint8Array {
  if (popcountTable) {
    return popcountTable;
  }
  const table = new Uint8Array(256);
  for (let i = 0; i < 256; i += 1) {
    let value = i;
    let count = 0;
    while (value > 0) {
      count += value & 1;
      value >>= 1;
    }
    table[i] = count;
  }
  popcountTable = table;
  return table;
}

function hammingDistance(a: Uint8Array, b: Uint8Array): number {
  const table = ensurePopcount();
  let total = 0;
  const length = Math.min(a.length, b.length);
  for (let i = 0; i < length; i += 1) {
    total += table[a[i] ^ b[i]];
  }
  return total;
}

function ensureScoringWorker(): Worker {
  if (scoringWorker) {
    return scoringWorker;
  }

  scoringWorker = new Worker(new URL("./scoring-worker.ts", import.meta.url), { type: "module" });
  scoringWorker.addEventListener("error", () => {
    scoringWorkerReady = false;
  });
  return scoringWorker;
}

function initializeScoringWorker(candidates: CandidateRow[]): void {
  const worker = ensureScoringWorker();
  const embeddings = candidates.map((candidate) => new Uint8Array(candidate.embedding));
  worker.postMessage({
    type: "init",
    embeddings,
  });
  scoringWorkerReady = true;
}

function scoreCandidatesSync(exemplars: CandidateRow[]): RankedRow[] {
  const results: RankedRow[] = [];
  for (const candidate of state.candidateRows) {
    if (exemplars.some((exemplar) => exemplar.chips_id === candidate.chips_id)) {
      continue;
    }
    const score =
      exemplars.reduce((sum, exemplar) => sum + hammingDistance(candidate.embedding, exemplar.embedding), 0) /
      exemplars.length;
    results.push({
      ...candidate,
      score,
    });
  }
  results.sort((a, b) => a.score - b.score || a.chips_id.localeCompare(b.chips_id));
  return results;
}

async function scoreCandidatesInWorker(exemplars: CandidateRow[]): Promise<RankedRow[]> {
  if (!scoringWorkerReady) {
    return scoreCandidatesSync(exemplars);
  }

  const worker = ensureScoringWorker();
  const requestId = ++scoringRequestId;
  const excludeIndices = new Set(exemplars.map((exemplar) => state.candidateRows.indexOf(exemplar)));
  const results = await new Promise<WorkerScoreResult[]>((resolve, reject) => {
    const handleMessage = (event: MessageEvent<{ type: string; requestId: number; results: WorkerScoreResult[] }>) => {
      if (event.data.type !== "score-result" || event.data.requestId !== requestId) {
        return;
      }
      worker.removeEventListener("message", handleMessage);
      worker.removeEventListener("error", handleError);
      resolve(event.data.results);
    };
    const handleError = (event: ErrorEvent) => {
      worker.removeEventListener("message", handleMessage);
      worker.removeEventListener("error", handleError);
      scoringWorkerReady = false;
      reject(event.error ?? new Error(event.message));
    };

    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);
    const exemplarBuffers = exemplars.map((exemplar) => new Uint8Array(exemplar.embedding));
    worker.postMessage({
      type: "score",
      requestId,
      exemplars: exemplarBuffers,
      excludeIndices: [...excludeIndices],
    });
  });

  return results.map(({ index, score }) => ({
    ...state.candidateRows[index],
    score,
  }));
}

function createAppShell(): void {
  const app = document.querySelector<HTMLDivElement>("#app");
  if (!app) {
    throw new Error("App root not found");
  }

  document.title = "terrabit binarized embeddings demo";
  app.innerHTML = `
    <main class="shell">
      <header class="app-header">
        <div class="app-brand">
          <h1>Terrabit: Binary Earth Embedding Retrieval</h1>
        </div>
      </header>

      <section class="workspace">
        <aside class="control-rail">
          <section class="rail-card rail-card-primary">
            <div class="rail-head">
              <div>
                <p class="panel-kicker">Search</p>
                <h2>Region query</h2>
              </div>
            </div>
            <div class="status-block">
              <span class="status-dot" aria-hidden="true"></span>
              <p id="status" class="status">${state.status}</p>
            </div>
            <div class="actions rail-actions">
              <button id="draw-aoi" class="primary" type="button">Draw region</button>
              <button id="clear-positives" class="ghost" type="button">Clear points</button>
            </div>
            <div class="shortcut-row" aria-label="interaction shortcuts">
              <span class="shortcut-chip">Shift drag: fast AOI</span>
              <span class="shortcut-chip">Esc: clear region</span>
            </div>
          </section>

          <section class="rail-card list-card">
            <div class="card-head">
              <div>
                <p class="panel-kicker">Positive points</p>
                <h3 class="card-title">Exemplars</h3>
              </div>
            </div>
            <ol id="positive-list" class="point-list"></ol>
          </section>
        </aside>

        <div class="map-panel">
          <div class="map-stage">
            <div class="map-legend" aria-hidden="true">
              <span><i class="legend-swatch legend-swatch-aoi"></i>AOI</span>
              <span><i class="legend-swatch legend-swatch-positive"></i>Positive</span>
              <span><i class="legend-swatch legend-swatch-rank"></i>Ranked</span>
            </div>
            <div id="map"></div>
          </div>
        </div>

        <aside class="sidebar">
          <section class="card list-card">
            <div class="card-head">
              <div>
                <p class="panel-kicker">Ranking</p>
                <h3 class="card-title">Candidate review</h3>
              </div>
              <span id="result-count">0 matches</span>
            </div>
            <div class="ranking-controls">
              <label class="range-control" for="topk-slider">
                <span>Top-k <strong id="topk-value">${DEFAULT_TOP_K}</strong></span>
                <input id="topk-slider" type="range" min="1" max="${MAX_TOP_K}" value="${DEFAULT_TOP_K}" />
              </label>
              <label class="toggle-control" for="heatmap-toggle">
                <span>Heatmap</span>
                <input id="heatmap-toggle" type="checkbox" />
              </label>
            </div>
            <ol id="result-list" class="result-list"></ol>
          </section>
        </aside>
      </section>
    </main>
  `;
}

function getElements() {
  return {
    status: document.querySelector<HTMLElement>("#status"),
    positiveList: document.querySelector<HTMLOListElement>("#positive-list"),
    resultCount: document.querySelector<HTMLElement>("#result-count"),
    resultList: document.querySelector<HTMLOListElement>("#result-list"),
    topkSlider: document.querySelector<HTMLInputElement>("#topk-slider"),
    topkValue: document.querySelector<HTMLElement>("#topk-value"),
    heatmapToggle: document.querySelector<HTMLInputElement>("#heatmap-toggle"),
    controlRail: document.querySelector<HTMLElement>(".control-rail"),
    drawAoi: document.querySelector<HTMLButtonElement>("#draw-aoi"),
    clearPositives: document.querySelector<HTMLButtonElement>("#clear-positives"),
  };
}

async function instantiateDuckDB(bundles: duckdb.DuckDBBundles): Promise<duckdb.AsyncDuckDB> {
  const bundle = await duckdb.selectBundle(bundles);
  if (!bundle.mainWorker) {
    throw new Error("DuckDB bundle is missing a worker URL");
  }
  const worker = new Worker(bundle.mainWorker);
  const logger = new duckdb.ConsoleLogger();
  const db = new duckdb.AsyncDuckDB(logger, worker);
  try {
    await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
    const conn = await db.connect();
    await conn.query("INSTALL httpfs; LOAD httpfs;");
    await conn.close();
    return db;
  } catch (error) {
    worker.terminate();
    throw error;
  }
}

function getDuckDB(): Promise<duckdb.AsyncDuckDB> {
  if (!dbPromise) {
    dbPromise = instantiateDuckDB(DUCKDB_BUNDLES);
  }
  return dbPromise;
}

function setStatus(message: string): void {
  state.status = message;
  updateView();
}

function updateView(): void {
  const els = getElements();
  if (!els.status || !els.positiveList || !els.resultCount || !els.resultList || !els.topkSlider || !els.topkValue || !els.heatmapToggle || !els.controlRail || !els.drawAoi) {
    return;
  }

  els.drawAoi.textContent = drawModeArmed ? "Drawing..." : state.bbox ? "Redraw region" : "Draw region";
  els.drawAoi.classList.toggle("is-armed", drawModeArmed);
  els.status.textContent = state.status;
  els.topkSlider.value = String(state.topK);
  els.topkValue.textContent = String(state.topK);
  els.heatmapToggle.checked = state.showHeatmap;
  const visibleResults = state.results.slice(0, state.topK);
  els.resultCount.textContent = state.showHeatmap
    ? `${new Intl.NumberFormat().format(state.results.length)} scored`
    : `${new Intl.NumberFormat().format(visibleResults.length)} shown of ${new Intl.NumberFormat().format(state.results.length)}`;

  els.positiveList.innerHTML = "";
  if (!state.positivePoints.length) {
    els.positiveList.innerHTML = `
      <li class="empty-state">
        No exemplars yet. Click inside the selected region to seed the search.
      </li>
    `;
  }
  for (const [index, point] of state.positivePoints.entries()) {
    const li = document.createElement("li");
    li.className = "list-item";
    li.style.setProperty("--item-index", String(index));
    li.innerHTML = `
      <button type="button" data-point-id="${point.id}" class="point-row">
        <span class="point-row-meta">
          <code>P-${point.id}</code>
          <strong>Positive sample</strong>
        </span>
        <span class="point-row-value">${point.lat.toFixed(5)}, ${point.lng.toFixed(5)}</span>
      </button>
    `;
    els.positiveList.appendChild(li);
  }

  els.positiveList.querySelectorAll<HTMLButtonElement>("button[data-point-id]").forEach((button) => {
    button.addEventListener("click", () => {
      const pointId = Number(button.dataset.pointId);
      state.positivePoints = state.positivePoints.filter((point) => point.id !== pointId);
      state.positivePoints = state.positivePoints.map((point, index) => ({ ...point, id: index + 1 }));
      void scoreCandidates();
      renderPositivePoints();
      updateView();
    });
  });

  els.resultList.innerHTML = "";
  if (!visibleResults.length) {
    const emptyMessage = !state.candidateRows.length
      ? "Load a region first to fetch candidate tiles."
      : !state.positivePoints.length
        ? "Add one or more positive points to produce a ranking."
        : "No ranked tiles yet.";
    els.resultList.innerHTML = `<li class="empty-state">${emptyMessage}</li>`;
  }
  for (const [index, result] of visibleResults.entries()) {
    const li = document.createElement("li");
    li.className = "list-item";
    li.style.setProperty("--item-index", String(index));
    const box = result.bbox;
    li.innerHTML = `
      <button data-chip="${result.chips_id}" class="result-row">
        <span class="result-row-top">
          <strong>Match ${index + 1}</strong>
          <code>${result.score.toFixed(1)}</code>
        </span>
        <span>${centroid(box).lat.toFixed(4)}, ${centroid(box).lng.toFixed(4)}</span>
      </button>
    `;
    els.resultList.appendChild(li);
  }

  els.resultList.querySelectorAll<HTMLButtonElement>("button[data-chip]").forEach((button) => {
    const chipId = button.dataset.chip;
    const row = state.results.find((item) => item.chips_id === chipId);
    if (!row) {
      return;
    }
    const clearPreview = (): void => {
      renderHoverPreview(null);
    };
    button.addEventListener("mouseenter", () => {
      renderHoverPreview(row);
      const map = (window as Window & { __terrabitMap?: any }).__terrabitMap;
      map?.panInsideBounds(
        [
          [row.bbox.south, row.bbox.west],
          [row.bbox.north, row.bbox.east],
        ],
        { animate: true, padding: [48, 48] },
      );
    });
    button.addEventListener("focus", () => {
      renderHoverPreview(row);
    });
    button.addEventListener("mouseleave", clearPreview);
    button.addEventListener("blur", clearPreview);
    button.addEventListener("click", () => {
      const center = centroid(row.bbox);
      const map = (window as Window & { __terrabitMap?: any }).__terrabitMap;
      if (map) {
        map.setView([center.lat, center.lng], Math.max(map.getZoom(), 11));
      }
    });
  });
}

async function queryManifestAndLoadCandidates(bbox: BBox): Promise<void> {
  state.bbox = bbox;
  state.positivePoints = [];
  state.results = [];
  state.candidateRows = [];
  state.manifestShards = [];
  state.shardCount = 0;
  state.candidateCount = 0;
  state.positiveMatches = [];
  clearLayers();
  updateView();

  setStatus("Loading intersecting shard list...");
  try {
    const db = await getDuckDB();
    const conn = await db.connect();
    const manifestQuery = buildManifestQuery(state.manifestUrl, bbox);
    const manifestResult = await conn.query(manifestQuery);
    const manifestRows = manifestResult.toArray() as ManifestRow[];
    await conn.close();

    state.manifestShards = manifestRows;
    state.shardCount = manifestRows.length;
    updateView();

    if (!manifestRows.length) {
      setStatus("No shards intersect that region.");
      return;
    }

    setStatus("Loading patch rows from intersecting shards...");
    const candidateConn = await db.connect();
    const candidateQuery = buildCandidateQuery(state.manifestUrl, manifestRows, bbox);
    const candidateResult = await candidateConn.query(candidateQuery);
    await candidateConn.close();

    const rows = candidateResult.toArray() as Array<{
      chips_id: string;
      bbox: { xmin: number; ymin: number; xmax: number; ymax: number };
      embedding: unknown;
      shard_path: string;
    }>;

    state.candidateRows = rows.map((row) => ({
      chips_id: row.chips_id,
      bbox: normalizeBBox(row.bbox),
      embedding: normalizeEmbedding(row.embedding),
      shard_path: row.shard_path,
    }));
    initializeScoringWorker(state.candidateRows);
    state.candidateCount = state.candidateRows.length;
    setStatus(
      state.candidateRows.length
        ? "Region loaded. Click positive points inside the box to rank similar patches."
        : "Region loaded, but no patch rows were returned.",
    );
    renderAoiBox(bbox);
    updateView();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(`Failed to load region data: ${message}`);
  }
}

async function scoreCandidates(): Promise<void> {
  const runId = ++latestScoreRunId;
  if (!state.candidateRows.length || !state.positivePoints.length) {
    state.positiveMatches = [];
    state.results = [];
    renderHoverPreview(null);
    renderPositiveMatches();
    renderResultsOnMap();
    updateView();
    return;
  }

  state.positiveMatches = resolvePositiveMatches();
  const exemplars = state.positiveMatches.map((match) => match.candidate);

  if (!exemplars.length) {
    state.results = [];
    renderHoverPreview(null);
    renderPositiveMatches();
    renderResultsOnMap();
    setStatus("No patch under one of the positive points. Click closer to a patch center.");
    return;
  }

  state.results = [];
  renderHoverPreview(null);
  renderPositiveMatches();
  renderResultsOnMap();
  updateView();
  setStatus(`Scoring ${new Intl.NumberFormat().format(state.candidateRows.length)} candidates...`);
  const scoredResults = await scoreCandidatesInWorker(exemplars);
  if (runId !== latestScoreRunId) {
    return;
  }
  state.results = scoredResults;
  renderResultsOnMap();
  setStatus(`Ranked ${state.candidateRows.length - exemplars.length} candidate patches against ${exemplars.length} positive patch(s).`);
}

function renderAoiBox(bbox: BBox): void {
  if (!aoiLayer) {
    return;
  }
  aoiLayer.clearLayers();
  L.rectangle(
    [
      [bbox.south, bbox.west],
      [bbox.north, bbox.east],
    ],
    {
      color: "#feefc3",
      weight: 2,
      fillOpacity: 0.08,
    },
  ).addTo(aoiLayer);
}

function clearLayers(): void {
  positiveLayer?.clearLayers();
  positiveMatchLayer?.clearLayers();
  resultLayer?.clearLayers();
  heatmapLayer?.clearLayers();
  previewLayer?.clearLayers();
  aoiLayer?.clearLayers();
  draftRectangle?.remove();
  draftRectangle = null;
  drawStartLatLng = null;
  drawMoved = false;
  if (mapRef) {
    mapRef.dragging.enable();
    mapRef.getContainer().style.cursor = "";
  }
}

function cancelAoiDraft(): void {
  draftRectangle?.remove();
  draftRectangle = null;
  drawStartLatLng = null;
  drawMoved = false;
  if (mapRef) {
    mapRef.dragging.enable();
    mapRef.getContainer().style.cursor = "";
  }
}

function clearPositiveSelection(status = "Positive points cleared."): void {
  state.positivePoints = [];
  state.positiveMatches = [];
  state.results = [];
  renderHoverPreview(null);
  renderPositivePoints();
  renderPositiveMatches();
  renderResultsOnMap();
  setStatus(status);
  updateView();
}

function clearRegion(status = "Region cleared."): void {
  state.bbox = null;
  state.controlsCollapsed = false;
  state.manifestShards = [];
  state.candidateRows = [];
  state.positivePoints = [];
  state.positiveMatches = [];
  state.results = [];
  state.shardCount = 0;
  state.candidateCount = 0;
  state.topK = DEFAULT_TOP_K;
  state.showHeatmap = false;
  drawModeArmed = false;
  clearLayers();
  setStatus(status);
  updateView();
}

function addPositivePoint(lat: number, lng: number, status?: string): void {
  const point: PositivePoint = {
    id: state.positivePoints.length + 1,
    lat,
    lng,
  };
  state.positivePoints.push(point);
  renderPositivePoints();
  void scoreCandidates();
  if (status) {
    setStatus(status);
  } else {
    updateView();
  }
}

function renderPositivePoints(): void {
  if (!positiveLayer) {
    return;
  }
  positiveLayer.clearLayers();
  for (const point of state.positivePoints) {
    L.circleMarker([point.lat, point.lng], {
      radius: 6,
      color: "#f25c54",
      fillColor: "#feefc3",
      fillOpacity: 1,
      weight: 2,
    }).addTo(positiveLayer);
  }
}

function renderPositiveMatches(): void {
  if (!positiveMatchLayer) {
    return;
  }
  positiveMatchLayer.clearLayers();
  for (const match of state.positiveMatches) {
    L.rectangle(
      [
        [match.candidate.bbox.south, match.candidate.bbox.west],
        [match.candidate.bbox.north, match.candidate.bbox.east],
      ],
      {
        color: "#6ef273",
        weight: 2,
        fillColor: "#6ef273",
        fillOpacity: 0.14,
      },
    )
      .bindPopup(`<strong>positive exemplar</strong><br />${match.candidate.chips_id}`)
      .addTo(positiveMatchLayer);
  }
}

function renderResultsOnMap(): void {
  if (!resultLayer || !heatmapLayer) {
    return;
  }
  resultLayer.clearLayers();
  heatmapLayer.clearLayers();

  const resultsToRender = state.showHeatmap ? state.results : state.results.slice(0, state.topK);
  if (!resultsToRender.length) {
    return;
  }

  const scores = resultsToRender.map((result) => result.score);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const scoreSpan = maxScore - minScore || 1;

  for (const result of resultsToRender) {
    const t = (result.score - minScore) / scoreSpan;
    const fillColor = state.showHeatmap ? interpolatePlasmaColor(t) : "#d9896a";
    const rectangle = L.rectangle(
      [
        [result.bbox.south, result.bbox.west],
        [result.bbox.north, result.bbox.east],
      ],
      {
        color: fillColor,
        fillColor,
        fillOpacity: state.showHeatmap ? 0.28 - t * 0.12 : 0.02,
        opacity: state.showHeatmap ? 0.88 - t * 0.28 : 0.9,
        renderer: state.showHeatmap ? heatmapRenderer : undefined,
        weight: state.showHeatmap ? 0.35 : 1,
      },
    );
    rectangle.on("click", (event: any) => {
      L.DomEvent.stopPropagation(event);
      const center = centroid(result.bbox);
      addPositivePoint(center.lat, center.lng, "Added match as positive exemplar.");
    });
    rectangle
      .bindPopup(`<strong>${result.chips_id}</strong><br />distance ${result.score.toFixed(1)}<br />click tile to add as exemplar`)
      .addTo(state.showHeatmap ? heatmapLayer : resultLayer);
  }
}

function interpolatePlasmaColor(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const stops = [
    { t: 0.0, r: 240, g: 249, b: 33 },
    { t: 0.25, r: 248, g: 149, b: 64 },
    { t: 0.5, r: 204, g: 71, b: 120 },
    { t: 0.75, r: 126, g: 3, b: 167 },
    { t: 1.0, r: 13, g: 8, b: 135 },
  ];
  for (let index = 0; index < stops.length - 1; index += 1) {
    const start = stops[index];
    const end = stops[index + 1];
    if (clamped >= start.t && clamped <= end.t) {
      const localT = (clamped - start.t) / (end.t - start.t);
      const mix = (a: number, b: number): number => Math.round(a + (b - a) * localT);
      return `rgb(${mix(start.r, end.r)} ${mix(start.g, end.g)} ${mix(start.b, end.b)})`;
    }
  }
  const last = stops[stops.length - 1];
  return `rgb(${last.r} ${last.g} ${last.b})`;
}

function renderHoverPreview(result: RankedRow | null): void {
  if (!previewLayer) {
    return;
  }
  previewLayer.clearLayers();
  if (!result) {
    return;
  }
  L.rectangle(
    [
      [result.bbox.south, result.bbox.west],
      [result.bbox.north, result.bbox.east],
    ],
    {
      color: "#82d4ca",
      weight: 2,
      fillColor: "#82d4ca",
      fillOpacity: 0.08,
      dashArray: "6 4",
      className: "hover-preview-bbox",
    },
  ).addTo(previewLayer);
}

function attachMap(): void {
  const map = L.map("map", {
    center: [20, 0],
    zoom: 2,
    zoomControl: false,
    boxZoom: false,
  });
  mapRef = map;
  (window as Window & { __terrabitMap?: any }).__terrabitMap = map;

  L.control.zoom({ position: "topright" }).addTo(map);
  requestAnimationFrame(() => map.invalidateSize());
  window.addEventListener("resize", () => map.invalidateSize());
  const mapElement = document.querySelector<HTMLElement>("#map");
  const resizeObserver = mapElement
    ? new ResizeObserver(() => {
        map.invalidateSize();
      })
    : null;
  if (mapElement && resizeObserver) {
    resizeObserver.observe(mapElement);
  }

  const sentinelLayer = L.tileLayer(SENTINEL_2024_TILES, {
    maxZoom: 14,
    attribution: SENTINEL_ATTRIBUTION,
    crossOrigin: true,
  });
  const osmLayer = L.tileLayer(OSM_TILES, {
    maxZoom: 19,
    attribution: OSM_ATTRIBUTION,
    crossOrigin: true,
  });

  sentinelLayer.on("tileerror", () => {
    if (!map.hasLayer(osmLayer)) {
      osmLayer.addTo(map);
    }
  });
  sentinelLayer.addTo(map);

  aoiLayer = new L.FeatureGroup().addTo(map);
  positiveLayer = new L.FeatureGroup().addTo(map);
  positiveMatchLayer = new L.FeatureGroup().addTo(map);
  resultLayer = new L.FeatureGroup().addTo(map);
  heatmapRenderer = L.canvas({ padding: 0.5 });
  heatmapLayer = new L.FeatureGroup().addTo(map);
  previewLayer = new L.FeatureGroup().addTo(map);

  const syncDraftRectangle = (endLatLng: { lat: number; lng: number }): void => {
    if (!drawStartLatLng) {
      return;
    }
    const south = Math.min(drawStartLatLng.lat, endLatLng.lat);
    const north = Math.max(drawStartLatLng.lat, endLatLng.lat);
    const west = Math.min(drawStartLatLng.lng, endLatLng.lng);
    const east = Math.max(drawStartLatLng.lng, endLatLng.lng);
    if (draftRectangle) {
      draftRectangle.setBounds([
        [south, west],
        [north, east],
      ]);
      return;
    }
    draftRectangle = L.rectangle(
      [
        [south, west],
        [north, east],
      ],
      {
        color: "#feefc3",
        weight: 2,
        fillOpacity: 0.08,
      },
    ).addTo(aoiLayer);
  };

  const startDrawing = (startLatLng: { lat: number; lng: number }): void => {
    if (!mapRef) {
      return;
    }
    cancelAoiDraft();
    drawStartLatLng = startLatLng;
    drawMoved = false;
    drawModeArmed = false;
    mapRef.dragging.disable();
    mapRef.getContainer().style.cursor = "crosshair";
    syncDraftRectangle(startLatLng);
    setStatus("Dragging region. Release to load matching patches.");
    updateView();
  };

  const finishDrawing = async (endLatLng: { lat: number; lng: number }): Promise<void> => {
    if (!mapRef || !drawStartLatLng) {
      return;
    }
    if (!drawMoved) {
      cancelAoiDraft();
      setStatus("Region draw canceled. Click Draw region or Shift-drag to try again.");
      return;
    }
    const south = Math.min(drawStartLatLng.lat, endLatLng.lat);
    const north = Math.max(drawStartLatLng.lat, endLatLng.lat);
    const west = Math.min(drawStartLatLng.lng, endLatLng.lng);
    const east = Math.max(drawStartLatLng.lng, endLatLng.lng);
    const bbox = { west, south, east, north };
    drawStartLatLng = null;
    drawMoved = false;
    draftRectangle?.remove();
    draftRectangle = L.rectangle(
      [
        [south, west],
        [north, east],
      ],
      {
        color: "#feefc3",
        weight: 2,
        fillOpacity: 0.08,
      },
    ).addTo(aoiLayer);
    mapRef.dragging.enable();
    mapRef.getContainer().style.cursor = "";
    suppressNextMapClick = true;
    await queryManifestAndLoadCandidates(bbox);
  };

  map.on("mousedown", (event: any) => {
    if (!event.originalEvent.shiftKey && !drawModeArmed) {
      return;
    }
    event.originalEvent.preventDefault();
    startDrawing(event.latlng);
  });

  map.on("mousemove", (event: any) => {
    if (!drawStartLatLng) {
      return;
    }
    drawMoved = true;
    syncDraftRectangle(event.latlng);
  });

  map.on("mouseup", (event: any) => {
    if (!drawStartLatLng) {
      return;
    }
    event.originalEvent.preventDefault();
    void finishDrawing(event.latlng);
  });

  map.on("click", (event: any) => {
    if (suppressNextMapClick) {
      suppressNextMapClick = false;
      return;
    }
    if (drawStartLatLng) {
      return;
    }
    if (!state.bbox || !containsPoint(state.bbox, event.latlng.lat, event.latlng.lng)) {
      return;
    }
    addPositivePoint(event.latlng.lat, event.latlng.lng);
  });

  getElements().drawAoi?.addEventListener("click", () => {
    if (state.bbox && !drawModeArmed) {
      clearRegion("Region cleared. Drag to draw a new region.");
      drawModeArmed = true;
      setStatus("Draw mode armed. Drag on the map to define a region.");
      map.getContainer().style.cursor = "crosshair";
      updateView();
      return;
    }
    drawModeArmed = !drawModeArmed;
    if (drawModeArmed) {
      setStatus("Draw mode armed. Drag on the map to define a region.");
      map.getContainer().style.cursor = "crosshair";
    } else {
      map.getContainer().style.cursor = "";
      setStatus(state.bbox ? "Region kept. Click inside it to add positive points." : "Draw mode off.");
    }
    updateView();
  });

  getElements().topkSlider?.addEventListener("input", (event) => {
    const target = event.currentTarget as HTMLInputElement;
    state.topK = Number(target.value);
    renderResultsOnMap();
    updateView();
  });

  getElements().heatmapToggle?.addEventListener("change", (event) => {
    const target = event.currentTarget as HTMLInputElement;
    state.showHeatmap = target.checked;
    renderResultsOnMap();
    updateView();
  });

  getElements().clearPositives?.addEventListener("click", () => {
    clearPositiveSelection();
  });

  window.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") {
      return;
    }
    if (drawStartLatLng) {
      cancelAoiDraft();
      setStatus("Region draw canceled.");
      updateView();
      return;
    }
    if (drawModeArmed) {
      drawModeArmed = false;
      map.getContainer().style.cursor = "";
      setStatus("Draw mode off.");
      updateView();
      return;
    }
    if (state.bbox) {
      clearRegion();
      return;
    }
    if (!state.positivePoints.length) {
      return;
    }
    clearPositiveSelection("Positive points cleared. Region kept.");
  });
}

createAppShell();
updateView();
attachMap();
