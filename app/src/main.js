import * as duckdb from "@duckdb/duckdb-wasm";
import duckdbWasmEh from "@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url";
import duckdbWasmMvp from "@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url";
import duckdbWorkerEh from "@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url";
import duckdbWorkerMvp from "@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url";
import * as L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./styles.css";
const MANIFEST_URL = "https://data.source.coop/geovibes/terrabit/clay-v1_5-binary-sentinel-2/manifest.parquet";
const DEFAULT_TOP_K = 10;
const MAX_MANIFEST_SHARDS = 256;
const MAX_AOI_ROWS = 50000;
const SENTINEL_2024_TILES = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg";
const OSM_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const SENTINEL_ATTRIBUTION = 'Sentinel-2 cloudless - <a href="https://s2maps.eu" target="_blank" rel="noreferrer">https://s2maps.eu</a> by <a href="https://eox.at" target="_blank" rel="noreferrer">EOX IT Services GmbH</a> (Contains modified Copernicus Sentinel data 2024)';
const OSM_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors';
const state = {
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
    isBusy: false,
};
let dbPromise = null;
let popcountTable = null;
let positiveLayer = null;
let positiveMatchLayer = null;
let aoiLayer = null;
let resultLayer = null;
let previewLayer = null;
let mapRef = null;
let drawStartLatLng = null;
let draftRectangle = null;
let drawMoved = false;
let drawModeArmed = false;
let scoringWorker = null;
let scoringWorkerReady = false;
let scoringRequestId = 0;
let latestScoreRunId = 0;
const DUCKDB_BUNDLES = {
    mvp: {
        mainModule: duckdbWasmMvp,
        mainWorker: duckdbWorkerMvp,
    },
    eh: {
        mainModule: duckdbWasmEh,
        mainWorker: duckdbWorkerEh,
    },
};
function getDefaultManifestUrl() {
    return MANIFEST_URL;
}
function sqlString(value) {
    return `'${value.replaceAll("'", "''")}'`;
}
function normalizeBBox(value) {
    const box = value;
    return {
        west: box.xmin,
        south: box.ymin,
        east: box.xmax,
        north: box.ymax,
    };
}
function normalizeEmbedding(value) {
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
        const candidate = value;
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
function overlaps(a, b) {
    return a.east >= b.west && a.west <= b.east && a.north >= b.south && a.south <= b.north;
}
function containsPoint(box, lat, lng) {
    return lng >= box.west && lng <= box.east && lat >= box.south && lat <= box.north;
}
function centroid(box) {
    return {
        lat: (box.south + box.north) / 2,
        lng: (box.west + box.east) / 2,
    };
}
function distanceSquared(aLat, aLng, bLat, bLng) {
    const dLat = aLat - bLat;
    const dLng = aLng - bLng;
    return dLat * dLat + dLng * dLng;
}
function resolvePositiveMatches() {
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
function buildManifestQuery(manifestUrl, bbox) {
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
function buildCandidateQuery(manifestUrl, shards, bbox) {
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
function resolveShardUrl(manifestUrl, relativePath) {
    try {
        return new URL(relativePath, manifestUrl).toString();
    }
    catch {
        return relativePath;
    }
}
function ensurePopcount() {
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
function hammingDistance(a, b) {
    const table = ensurePopcount();
    let total = 0;
    const length = Math.min(a.length, b.length);
    for (let i = 0; i < length; i += 1) {
        total += table[a[i] ^ b[i]];
    }
    return total;
}
function ensureScoringWorker() {
    if (scoringWorker) {
        return scoringWorker;
    }
    scoringWorker = new Worker(new URL("./scoring-worker.ts", import.meta.url), { type: "module" });
    scoringWorker.addEventListener("error", () => {
        scoringWorkerReady = false;
    });
    return scoringWorker;
}
function initializeScoringWorker(candidates) {
    const worker = ensureScoringWorker();
    const embeddings = candidates.map((candidate) => new Uint8Array(candidate.embedding));
    worker.postMessage({
        type: "init",
        embeddings,
    });
    scoringWorkerReady = true;
}
function scoreCandidatesSync(exemplars) {
    const results = [];
    for (const candidate of state.candidateRows) {
        if (exemplars.some((exemplar) => exemplar.chips_id === candidate.chips_id)) {
            continue;
        }
        const score = exemplars.reduce((sum, exemplar) => sum + hammingDistance(candidate.embedding, exemplar.embedding), 0) /
            exemplars.length;
        results.push({
            ...candidate,
            score,
        });
    }
    results.sort((a, b) => a.score - b.score || a.chips_id.localeCompare(b.chips_id));
    return results.slice(0, DEFAULT_TOP_K);
}
async function scoreCandidatesInWorker(exemplars) {
    if (!scoringWorkerReady) {
        return scoreCandidatesSync(exemplars);
    }
    const worker = ensureScoringWorker();
    const requestId = ++scoringRequestId;
    const excludeIndices = new Set(exemplars.map((exemplar) => state.candidateRows.indexOf(exemplar)));
    const results = await new Promise((resolve, reject) => {
        const handleMessage = (event) => {
            if (event.data.type !== "score-result" || event.data.requestId !== requestId) {
                return;
            }
            worker.removeEventListener("message", handleMessage);
            worker.removeEventListener("error", handleError);
            resolve(event.data.results);
        };
        const handleError = (event) => {
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
            topK: DEFAULT_TOP_K,
        });
    });
    return results.map(({ index, score }) => ({
        ...state.candidateRows[index],
        score,
    }));
}
function createAppShell() {
    const app = document.querySelector("#app");
    if (!app) {
        throw new Error("App root not found");
    }
    document.title = "terrabit binarized embeddings demo";
    app.innerHTML = `
    <main class="shell">
      <section class="workspace">
        <aside class="control-rail">
          <section class="rail-card rail-card-primary">
            <div class="rail-head">
              <div>
                <p class="panel-kicker">Demo</p>
                <h2>terrabit binarized embeddings demo</h2>
              </div>
              <button id="controls-toggle" class="ghost rail-toggle" type="button" aria-expanded="true">Hide</button>
            </div>
            <p class="hint hint-rail">Draw a region with the button or Shift-drag. Click inside for positives. Escape clears points.</p>
            <p id="status" class="status">${state.status}</p>
            <div class="actions rail-actions">
              <button id="draw-aoi" class="primary" type="button">Draw region</button>
              <button id="rerun-search" class="primary" type="button">Run search</button>
              <button id="clear-positives" class="ghost" type="button">Clear points</button>
              <button id="clear-aoi" class="ghost" type="button">Clear region</button>
            </div>
          </section>

          <section class="rail-card rail-meta">
            <p class="panel-kicker">Footprint</p>
            <dl class="stats">
              <div>
                <dt>Region shards</dt>
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
        </aside>

        <div class="map-panel">
          <div class="map-topbar">
            <div>
              <p class="panel-kicker">Region canvas</p>
              <h2>Map</h2>
            </div>
            <p class="hint">Shift-drag still works. Use the left rail when you want explicit controls.</p>
          </div>
          <div class="map-stage">
            <div id="map"></div>
          </div>
        </div>

        <aside class="sidebar">
          <section class="card list-card">
            <p class="panel-kicker">Positive points</p>
            <ol id="positive-list" class="point-list"></ol>
          </section>

          <section class="card list-card">
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
        status: document.querySelector("#status"),
        shardCount: document.querySelector("#shard-count"),
        candidateCount: document.querySelector("#candidate-count"),
        positiveCount: document.querySelector("#positive-count"),
        positiveList: document.querySelector("#positive-list"),
        resultCount: document.querySelector("#result-count"),
        resultList: document.querySelector("#result-list"),
        controlRail: document.querySelector(".control-rail"),
        controlsToggle: document.querySelector("#controls-toggle"),
        drawAoi: document.querySelector("#draw-aoi"),
        rerunSearch: document.querySelector("#rerun-search"),
        clearPositives: document.querySelector("#clear-positives"),
        clearAoi: document.querySelector("#clear-aoi"),
    };
}
async function instantiateDuckDB(bundles) {
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
    }
    catch (error) {
        worker.terminate();
        throw error;
    }
}
function getDuckDB() {
    if (!dbPromise) {
        dbPromise = instantiateDuckDB(DUCKDB_BUNDLES);
    }
    return dbPromise;
}
function setStatus(message) {
    state.status = message;
    updateView();
}
function updateView() {
    const els = getElements();
    if (!els.status || !els.shardCount || !els.candidateCount || !els.positiveCount || !els.positiveList || !els.resultCount || !els.resultList || !els.controlRail || !els.controlsToggle || !els.drawAoi) {
        return;
    }
    els.controlRail.classList.toggle("is-collapsed", state.controlsCollapsed);
    els.controlsToggle.textContent = state.controlsCollapsed ? "Show" : "Hide";
    els.controlsToggle.setAttribute("aria-expanded", String(!state.controlsCollapsed));
    els.drawAoi.textContent = drawModeArmed ? "Drawing..." : "Draw region";
    els.drawAoi.classList.toggle("is-armed", drawModeArmed);
    els.status.textContent = state.status;
    els.shardCount.textContent = new Intl.NumberFormat().format(state.shardCount);
    els.candidateCount.textContent = new Intl.NumberFormat().format(state.candidateCount);
    els.positiveCount.textContent = new Intl.NumberFormat().format(state.positivePoints.length);
    els.resultCount.textContent = `${new Intl.NumberFormat().format(state.results.length)} matches`;
    els.positiveList.innerHTML = "";
    for (const [index, point] of state.positivePoints.entries()) {
        const li = document.createElement("li");
        li.className = "list-item";
        li.style.setProperty("--item-index", String(index));
        li.innerHTML = `
      <button type="button" data-point-id="${point.id}" class="point-row">
        <code>#${point.id}</code>
        <span>${point.lat.toFixed(5)}, ${point.lng.toFixed(5)}</span>
      </button>
    `;
        els.positiveList.appendChild(li);
    }
    els.positiveList.querySelectorAll("button[data-point-id]").forEach((button) => {
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
    for (const [index, result] of state.results.slice(0, DEFAULT_TOP_K).entries()) {
        const li = document.createElement("li");
        li.className = "list-item";
        li.style.setProperty("--item-index", String(index));
        const box = result.bbox;
        li.innerHTML = `
      <button data-chip="${result.chips_id}" class="result-row">
        <strong>${result.chips_id}</strong>
        <span>score ${result.score.toFixed(1)} | ${centroid(box).lat.toFixed(4)}, ${centroid(box).lng.toFixed(4)}</span>
      </button>
    `;
        els.resultList.appendChild(li);
    }
    els.resultList.querySelectorAll("button[data-chip]").forEach((button) => {
        const chipId = button.dataset.chip;
        const row = state.results.find((item) => item.chips_id === chipId);
        if (!row) {
            return;
        }
        const clearPreview = () => {
            renderHoverPreview(null);
        };
        button.addEventListener("mouseenter", () => {
            renderHoverPreview(row);
            const map = window.__terrabitMap;
            map?.panInsideBounds([
                [row.bbox.south, row.bbox.west],
                [row.bbox.north, row.bbox.east],
            ], { animate: true, padding: [48, 48] });
        });
        button.addEventListener("focus", () => {
            renderHoverPreview(row);
        });
        button.addEventListener("mouseleave", clearPreview);
        button.addEventListener("blur", clearPreview);
        button.addEventListener("click", () => {
            const center = centroid(row.bbox);
            const map = window.__terrabitMap;
            if (map) {
                map.setView([center.lat, center.lng], Math.max(map.getZoom(), 11));
            }
        });
    });
}
async function queryManifestAndLoadCandidates(bbox) {
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
        const manifestRows = manifestResult.toArray();
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
        const rows = candidateResult.toArray();
        state.candidateRows = rows.map((row) => ({
            chips_id: row.chips_id,
            bbox: normalizeBBox(row.bbox),
            embedding: normalizeEmbedding(row.embedding),
            shard_path: row.shard_path,
        }));
        initializeScoringWorker(state.candidateRows);
        state.candidateCount = state.candidateRows.length;
        setStatus(state.candidateRows.length
            ? "Region loaded. Click positive points inside the box to rank similar patches."
            : "Region loaded, but no patch rows were returned.");
        renderAoiBox(bbox);
        updateView();
    }
    catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setStatus(`Failed to load region data: ${message}`);
    }
}
async function scoreCandidates() {
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
function renderAoiBox(bbox) {
    if (!aoiLayer) {
        return;
    }
    aoiLayer.clearLayers();
    L.rectangle([
        [bbox.south, bbox.west],
        [bbox.north, bbox.east],
    ], {
        color: "#feefc3",
        weight: 2,
        fillOpacity: 0.08,
    }).addTo(aoiLayer);
}
function clearLayers() {
    positiveLayer?.clearLayers();
    positiveMatchLayer?.clearLayers();
    resultLayer?.clearLayers();
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
function cancelAoiDraft() {
    draftRectangle?.remove();
    draftRectangle = null;
    drawStartLatLng = null;
    drawMoved = false;
    if (mapRef) {
        mapRef.dragging.enable();
        mapRef.getContainer().style.cursor = "";
    }
}
function clearPositiveSelection(status = "Positive points cleared.") {
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
function renderPositivePoints() {
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
function renderPositiveMatches() {
    if (!positiveMatchLayer) {
        return;
    }
    positiveMatchLayer.clearLayers();
    for (const match of state.positiveMatches) {
        L.rectangle([
            [match.candidate.bbox.south, match.candidate.bbox.west],
            [match.candidate.bbox.north, match.candidate.bbox.east],
        ], {
            color: "#6ef273",
            weight: 2,
            fillColor: "#6ef273",
            fillOpacity: 0.14,
        })
            .bindPopup(`<strong>positive exemplar</strong><br />${match.candidate.chips_id}`)
            .addTo(positiveMatchLayer);
    }
}
function renderResultsOnMap() {
    if (!resultLayer) {
        return;
    }
    resultLayer.clearLayers();
    for (const result of state.results) {
        L.rectangle([
            [result.bbox.south, result.bbox.west],
            [result.bbox.north, result.bbox.east],
        ], {
            color: "#f25c54",
            weight: 1,
            fillOpacity: 0.02,
        })
            .bindPopup(`<strong>${result.chips_id}</strong><br />score ${result.score.toFixed(1)}`)
            .addTo(resultLayer);
    }
}
function renderHoverPreview(result) {
    if (!previewLayer) {
        return;
    }
    previewLayer.clearLayers();
    if (!result) {
        return;
    }
    L.rectangle([
        [result.bbox.south, result.bbox.west],
        [result.bbox.north, result.bbox.east],
    ], {
        color: "#82d4ca",
        weight: 2,
        fillColor: "#82d4ca",
        fillOpacity: 0.08,
        dashArray: "6 4",
        className: "hover-preview-bbox",
    }).addTo(previewLayer);
}
function attachMap() {
    const map = L.map("map", {
        center: [20, 0],
        zoom: 2,
        zoomControl: false,
        boxZoom: false,
    });
    mapRef = map;
    window.__terrabitMap = map;
    L.control.zoom({ position: "topright" }).addTo(map);
    requestAnimationFrame(() => map.invalidateSize());
    window.addEventListener("resize", () => map.invalidateSize());
    const mapElement = document.querySelector("#map");
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
    previewLayer = new L.FeatureGroup().addTo(map);
    const syncDraftRectangle = (endLatLng) => {
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
        draftRectangle = L.rectangle([
            [south, west],
            [north, east],
        ], {
            color: "#feefc3",
            weight: 2,
            fillOpacity: 0.08,
        }).addTo(aoiLayer);
    };
    const startDrawing = (startLatLng) => {
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
    const finishDrawing = async (endLatLng) => {
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
        draftRectangle = L.rectangle([
            [south, west],
            [north, east],
        ], {
            color: "#feefc3",
            weight: 2,
            fillOpacity: 0.08,
        }).addTo(aoiLayer);
        mapRef.dragging.enable();
        mapRef.getContainer().style.cursor = "";
        await queryManifestAndLoadCandidates(bbox);
    };
    map.on("mousedown", (event) => {
        if (!event.originalEvent.shiftKey && !drawModeArmed) {
            return;
        }
        event.originalEvent.preventDefault();
        startDrawing(event.latlng);
    });
    map.on("mousemove", (event) => {
        if (!drawStartLatLng) {
            return;
        }
        drawMoved = true;
        syncDraftRectangle(event.latlng);
    });
    map.on("mouseup", (event) => {
        if (!drawStartLatLng) {
            return;
        }
        event.originalEvent.preventDefault();
        void finishDrawing(event.latlng);
    });
    map.on("click", (event) => {
        if (drawStartLatLng) {
            return;
        }
        if (!state.bbox || !containsPoint(state.bbox, event.latlng.lat, event.latlng.lng)) {
            return;
        }
        const point = {
            id: state.positivePoints.length + 1,
            lat: event.latlng.lat,
            lng: event.latlng.lng,
        };
        state.positivePoints.push(point);
        renderPositivePoints();
        void scoreCandidates();
        updateView();
    });
    const rerunSearch = getElements().rerunSearch;
    getElements().drawAoi?.addEventListener("click", () => {
        drawModeArmed = !drawModeArmed;
        if (drawModeArmed) {
            setStatus("Draw mode armed. Drag on the map to define a region.");
            map.getContainer().style.cursor = "crosshair";
        }
        else {
            map.getContainer().style.cursor = "";
            setStatus(state.bbox ? "Region kept. Click inside it to add positive points." : "Draw mode off.");
        }
        updateView();
    });
    rerunSearch?.addEventListener("click", () => {
        void scoreCandidates();
        renderResultsOnMap();
        updateView();
    });
    getElements().clearPositives?.addEventListener("click", () => {
        clearPositiveSelection();
    });
    getElements().clearAoi?.addEventListener("click", () => {
        state.bbox = null;
        state.controlsCollapsed = false;
        state.manifestShards = [];
        state.candidateRows = [];
        state.positivePoints = [];
        state.positiveMatches = [];
        state.results = [];
        state.shardCount = 0;
        state.candidateCount = 0;
        drawModeArmed = false;
        clearLayers();
        setStatus("Region cleared.");
        updateView();
    });
    getElements().controlsToggle?.addEventListener("click", () => {
        state.controlsCollapsed = !state.controlsCollapsed;
        requestAnimationFrame(() => map.invalidateSize());
        updateView();
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
        if (!state.positivePoints.length) {
            return;
        }
        clearPositiveSelection("Positive points cleared. Region kept.");
    });
}
createAppShell();
updateView();
attachMap();
