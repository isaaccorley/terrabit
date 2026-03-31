import * as duckdb from "@duckdb/duckdb-wasm";
import duckdbWasmEh from "@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url";
import duckdbWasmMvp from "@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url";
import duckdbWorkerEh from "@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url";
import duckdbWorkerMvp from "@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url";
import * as L from "leaflet";
import "leaflet/dist/leaflet.css";

import "./styles.css";

const MANIFEST_URL =
  "https://data.source.coop/geovibes/terrabit/clay-v1_5-binary/manifest.parquet";
const DEFAULT_TOP_K = 10;
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
  manifestShards: ManifestRow[];
  candidateRows: CandidateRow[];
  positivePoints: PositivePoint[];
  results: RankedRow[];
  shardCount: number;
  candidateCount: number;
  isBusy: boolean;
};

const state: AppState = {
  manifestUrl: getDefaultManifestUrl(),
  bbox: null,
  status: "Draw an AOI box, then click positive points inside it.",
  manifestShards: [],
  candidateRows: [],
  positivePoints: [],
  results: [],
  shardCount: 0,
  candidateCount: 0,
  isBusy: false,
};

let dbPromise: Promise<duckdb.AsyncDuckDB> | null = null;
let popcountTable: Uint8Array | null = null;
let positiveLayer: any = null;
let aoiLayer: any = null;
let resultLayer: any = null;
let mapRef: any = null;
let drawingEnabled = false;
let drawStartLatLng: { lat: number; lng: number } | null = null;
let draftRectangle: any = null;

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

function createAppShell(): void {
  const app = document.querySelector<HTMLDivElement>("#app");
  if (!app) {
    throw new Error("App root not found");
  }

  app.innerHTML = `
    <main class="shell">
      <section class="workspace">
        <div class="map-panel">
          <div class="map-topbar">
            <div>
              <p class="panel-kicker">AOI canvas</p>
              <h2>Sentinel-2 cloudless base</h2>
            </div>
            <p class="hint">Draw a rectangle, then click inside it to add positive examples.</p>
          </div>
          <div id="map"></div>
        </div>

        <aside class="sidebar">
          <section class="card hero-card">
            <p class="eyebrow">TerraBit browser test</p>
            <h1>Draw an AOI, click points.</h1>
            <p class="lede">
              Client-side ranking over hosted patch embeddings.
            </p>
            <div class="actions actions-compact">
              <button id="draw-aoi" class="primary">Draw AOI</button>
              <button id="rerun-search" class="primary">Run search</button>
              <button id="clear-positives" class="ghost">Clear points</button>
              <button id="clear-aoi" class="ghost">Clear AOI</button>
            </div>
          </section>

          <section class="card">
            <p class="panel-kicker">Status</p>
            <p id="status" class="status">${state.status}</p>
            <dl class="stats">
              <div>
                <dt>AOI shards</dt>
                <dd id="shard-count">0</dd>
              </div>
              <div>
                <dt>Loaded patches</dt>
                <dd id="candidate-count">0</dd>
              </div>
              <div>
                <dt>Positive points</dt>
                <dd id="positive-count">0</dd>
              </div>
            </dl>
          </section>

          <section class="card">
            <p class="panel-kicker">Positive points</p>
            <ol id="positive-list" class="point-list"></ol>
          </section>

          <section class="card">
            <div class="card-head">
              <p class="panel-kicker">Top-k</p>
              <span id="result-count">0 matches</span>
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
    shardCount: document.querySelector<HTMLElement>("#shard-count"),
    candidateCount: document.querySelector<HTMLElement>("#candidate-count"),
    positiveCount: document.querySelector<HTMLElement>("#positive-count"),
    positiveList: document.querySelector<HTMLOListElement>("#positive-list"),
    resultCount: document.querySelector<HTMLElement>("#result-count"),
    resultList: document.querySelector<HTMLOListElement>("#result-list"),
    drawAoi: document.querySelector<HTMLButtonElement>("#draw-aoi"),
    rerunSearch: document.querySelector<HTMLButtonElement>("#rerun-search"),
    clearPositives: document.querySelector<HTMLButtonElement>("#clear-positives"),
    clearAoi: document.querySelector<HTMLButtonElement>("#clear-aoi"),
  };
}

function getDuckDB(): Promise<duckdb.AsyncDuckDB> {
  if (!dbPromise) {
    dbPromise = (async () => {
      const bundle = await duckdb.selectBundle(DUCKDB_BUNDLES);
      if (!bundle.mainWorker) {
        throw new Error("DuckDB bundle is missing a worker URL");
      }
      const worker = new Worker(bundle.mainWorker);
      const logger = new duckdb.ConsoleLogger();
      const db = new duckdb.AsyncDuckDB(logger, worker);
      await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
      const conn = await db.connect();
      await conn.query("INSTALL httpfs; LOAD httpfs;");
      await conn.close();
      return db;
    })();
  }
  return dbPromise;
}

function setStatus(message: string): void {
  state.status = message;
  updateView();
}

function updateView(): void {
  const els = getElements();
  if (!els.status || !els.shardCount || !els.candidateCount || !els.positiveCount || !els.positiveList || !els.resultCount || !els.resultList) {
    return;
  }

  els.status.textContent = state.status;
  els.shardCount.textContent = new Intl.NumberFormat().format(state.shardCount);
  els.candidateCount.textContent = new Intl.NumberFormat().format(state.candidateCount);
  els.positiveCount.textContent = new Intl.NumberFormat().format(state.positivePoints.length);
  els.resultCount.textContent = `${new Intl.NumberFormat().format(state.results.length)} matches`;

  els.positiveList.innerHTML = "";
  for (const point of state.positivePoints) {
    const li = document.createElement("li");
    li.innerHTML = `<code>#${point.id}</code><span>${point.lat.toFixed(5)}, ${point.lng.toFixed(5)}</span>`;
    els.positiveList.appendChild(li);
  }

  els.resultList.innerHTML = "";
  for (const result of state.results.slice(0, DEFAULT_TOP_K)) {
    const li = document.createElement("li");
    const box = result.bbox;
    li.innerHTML = `
      <button data-chip="${result.chips_id}" class="result-row">
        <strong>${result.chips_id}</strong>
        <span>score ${result.score.toFixed(1)} | ${centroid(box).lat.toFixed(4)}, ${centroid(box).lng.toFixed(4)}</span>
      </button>
    `;
    els.resultList.appendChild(li);
  }

  els.resultList.querySelectorAll<HTMLButtonElement>("button[data-chip]").forEach((button) => {
    button.addEventListener("click", () => {
      const chipId = button.dataset.chip;
      const row = state.results.find((item) => item.chips_id === chipId);
      if (!row) {
        return;
      }
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
      setStatus("No shards intersect that AOI.");
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
    state.candidateCount = state.candidateRows.length;
    setStatus(
      state.candidateRows.length
        ? "AOI loaded. Click positive points inside the box to rank similar patches."
        : "AOI loaded, but no patch rows were returned.",
    );
    renderAoiBox(bbox);
    updateView();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(`Failed to load AOI data: ${message}`);
  }
}

function scoreCandidates(): void {
  if (!state.candidateRows.length || !state.positivePoints.length) {
    state.results = [];
    renderResultsOnMap();
    updateView();
    return;
  }

  const exemplars = state.positivePoints
    .map((point) => state.candidateRows.find((candidate) => containsPoint(candidate.bbox, point.lat, point.lng)))
    .filter((item): item is CandidateRow => Boolean(item));

  if (!exemplars.length) {
    state.results = [];
    renderResultsOnMap();
    setStatus("No patch under one of the positive points. Click closer to a patch center.");
    return;
  }

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
  state.results = results.slice(0, DEFAULT_TOP_K);
  renderResultsOnMap();
  setStatus(`Ranked ${results.length} candidate patches against ${exemplars.length} positive patch(s).`);
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
  resultLayer?.clearLayers();
  aoiLayer?.clearLayers();
  draftRectangle?.remove();
  draftRectangle = null;
  drawStartLatLng = null;
  drawingEnabled = false;
  if (mapRef) {
    mapRef.dragging.enable();
    mapRef.getContainer().style.cursor = "";
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

function renderResultsOnMap(): void {
  if (!resultLayer) {
    return;
  }
  resultLayer.clearLayers();
  for (const result of state.results) {
    L.rectangle(
      [
        [result.bbox.south, result.bbox.west],
        [result.bbox.north, result.bbox.east],
      ],
      {
        color: "#f25c54",
        weight: 1,
        fillOpacity: 0.02,
      },
    )
      .bindPopup(`<strong>${result.chips_id}</strong><br />score ${result.score.toFixed(1)}`)
      .addTo(resultLayer);
  }
}

function attachMap(): void {
  const map = L.map("map", {
    center: [20, 0],
    zoom: 2,
    zoomControl: false,
  });
  mapRef = map;
  (window as Window & { __terrabitMap?: any }).__terrabitMap = map;

  L.control.zoom({ position: "topright" }).addTo(map);

  const sentinelLayer = L.tileLayer(SENTINEL_2024_TILES, {
    maxZoom: 14,
    attribution: SENTINEL_ATTRIBUTION,
  });
  const osmLayer = L.tileLayer(OSM_TILES, {
    maxZoom: 19,
    attribution: OSM_ATTRIBUTION,
  });

  sentinelLayer.on("tileerror", () => {
    if (!map.hasLayer(osmLayer)) {
      osmLayer.addTo(map);
    }
  });
  sentinelLayer.addTo(map);

  aoiLayer = new L.FeatureGroup().addTo(map);
  positiveLayer = new L.FeatureGroup().addTo(map);
  resultLayer = new L.FeatureGroup().addTo(map);

  const startDrawing = (): void => {
    if (!mapRef) {
      return;
    }
    drawingEnabled = true;
    drawStartLatLng = null;
    draftRectangle?.remove();
    draftRectangle = null;
    mapRef.dragging.disable();
    mapRef.getContainer().style.cursor = "crosshair";
    setStatus("Click two opposite corners to draw an AOI box.");
  };

  const finishDrawing = async (endLatLng: { lat: number; lng: number }): Promise<void> => {
    if (!mapRef || !drawStartLatLng) {
      return;
    }
    const south = Math.min(drawStartLatLng.lat, endLatLng.lat);
    const north = Math.max(drawStartLatLng.lat, endLatLng.lat);
    const west = Math.min(drawStartLatLng.lng, endLatLng.lng);
    const east = Math.max(drawStartLatLng.lng, endLatLng.lng);
    const bbox = { west, south, east, north };
    drawingEnabled = false;
    drawStartLatLng = null;
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
    await queryManifestAndLoadCandidates(bbox);
  };

  map.on("click", (event: any) => {
    if (drawingEnabled) {
      if (!drawStartLatLng) {
        drawStartLatLng = event.latlng;
        if (draftRectangle) {
          draftRectangle.remove();
        }
        draftRectangle = L.rectangle(
          [
            [event.latlng.lat, event.latlng.lng],
            [event.latlng.lat, event.latlng.lng],
          ],
          {
            color: "#feefc3",
            weight: 2,
            fillOpacity: 0.08,
          },
        ).addTo(aoiLayer);
        setStatus("Click the opposite corner to finish the AOI box.");
      } else {
        void finishDrawing(event.latlng);
      }
      return;
    }

    if (!state.bbox || !containsPoint(state.bbox, event.latlng.lat, event.latlng.lng)) {
      return;
    }
    const point: PositivePoint = {
      id: state.positivePoints.length + 1,
      lat: event.latlng.lat,
      lng: event.latlng.lng,
    };
    state.positivePoints.push(point);
    renderPositivePoints();
    scoreCandidates();
    renderResultsOnMap();
    updateView();
  });

  getElements().drawAoi?.addEventListener("click", () => {
    startDrawing();
  });

  const rerunSearch = getElements().rerunSearch;
  rerunSearch?.addEventListener("click", () => {
    scoreCandidates();
    renderResultsOnMap();
    updateView();
  });

  getElements().clearPositives?.addEventListener("click", () => {
    state.positivePoints = [];
    state.results = [];
    renderPositivePoints();
    renderResultsOnMap();
    setStatus("Positive points cleared.");
  });

  getElements().clearAoi?.addEventListener("click", () => {
    state.bbox = null;
    state.manifestShards = [];
    state.candidateRows = [];
    state.positivePoints = [];
    state.results = [];
    state.shardCount = 0;
    state.candidateCount = 0;
    clearLayers();
    setStatus("AOI cleared.");
    updateView();
  });
}

createAppShell();
updateView();
attachMap();
