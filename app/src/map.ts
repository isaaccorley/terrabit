// The minified MapLibre bundle broke GeoJSON rendering in our static prod build.
// Use the dev bundle until upstream bundling/minification is safe here again.
import maplibregl from "maplibre-gl/dist/maplibre-gl-dev.js";
import "maplibre-gl/dist/maplibre-gl.css";

import type {
  AoiEntry,
  BBox,
  NegativePoint,
  PositiveMatch,
  PositivePoint,
  RankedRow,
  ViewMode,
} from "./types";
import { centroid, interpolatePlasma } from "./util";

const SENTINEL_TILES =
  "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg";
const SENTINEL_ATTRIBUTION =
  'Sentinel-2 cloudless — <a href="https://s2maps.eu" target="_blank" rel="noreferrer">s2maps.eu</a> by <a href="https://eox.at" target="_blank" rel="noreferrer">EOX</a> (Copernicus Sentinel data 2024)';

export type MapCallbacks = {
  onDrawComplete: (result: { bbox: BBox; polygon?: [number, number][] }) => void;
  onAoiClick: (lat: number, lng: number) => void;
  onNegativeClick: (lat: number, lng: number) => void;
  onResultHover: (result: RankedRow | null) => void;
  onResultPick: (result: RankedRow) => void;
  getBBox: () => BBox | null;
  getResults: () => RankedRow[];
  getTopK: () => number;
};

type DrawMode = "rect" | "polygon";

type DrawState = {
  mode: DrawMode;
  startLngLat: maplibregl.LngLat | null;
  startPoint: { x: number; y: number } | null;
  moved: boolean;
  armed: boolean;
  box: HTMLDivElement | null;
  polyVertices: maplibregl.LngLat[];
  polyActive: boolean;
};

export class GlobeMap {
  readonly map: maplibregl.Map;
  private cb: MapCallbacks;
  private _lastResultsKey = "";
  private draw: DrawState = {
    mode: "rect",
    startLngLat: null,
    startPoint: null,
    moved: false,
    armed: false,
    box: null,
    polyVertices: [],
    polyActive: false,
  };
  private styleReady = false;
  private pendingRender: (() => void)[] = [];

  constructor(container: HTMLElement, cb: MapCallbacks) {
    this.cb = cb;
    this.map = new maplibregl.Map({
      container,
      style: this.buildStyle(),
      center: [-95, 38],
      zoom: 1.8,
      minZoom: 0.5,
      maxZoom: 14,
      attributionControl: false,
      dragRotate: false,
      pitchWithRotate: false,
      touchZoomRotate: true,
      renderWorldCopies: false,
    });

    // MapLibre's default boxZoom handler eats shift+drag. We use shift+drag
    // for AOI drawing, so disable it here.
    this.map.boxZoom.disable();

    this.map.addControl(
      new maplibregl.NavigationControl({ showCompass: false, visualizePitch: false }),
      "top-right",
    );

    this.map.on("load", () => {
      try {
        this.map.setProjection({ type: "globe" });
      } catch {
        /* older builds */
      }
      this.addSources();
      this.addLayers();
      this.styleReady = true;
      this.map.resize();
      this.easeIntro();
      for (const fn of this.pendingRender.splice(0)) fn();
    });

    // Keep the map sized to its container — catches late layout shifts.
    const ro = new ResizeObserver(() => this.map.resize());
    ro.observe(container);
    window.addEventListener("resize", () => this.map.resize());

    this.wireDrawing();
    this.wireClicks();
  }

  private buildStyle(): maplibregl.StyleSpecification {
    return {
      version: 8,
      projection: { type: "globe" },
      glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
      sources: {
        sentinel: {
          type: "raster",
          tiles: [SENTINEL_TILES],
          tileSize: 256,
          minzoom: 0,
          maxzoom: 14,
          attribution: SENTINEL_ATTRIBUTION,
        },
      },
      layers: [
        {
          id: "bg",
          type: "background",
          paint: { "background-color": "#1a1612" },
        },
        {
          id: "sentinel",
          type: "raster",
          source: "sentinel",
          paint: {
            "raster-opacity": 1,
            "raster-fade-duration": 260,
          },
        },
      ],
      sky: {
        "sky-color": "#2b1a10",
        "horizon-color": "#6b3a1c",
        "fog-color": "#1a1612",
        "fog-ground-blend": 0.5,
        "horizon-fog-blend": 0.5,
        "sky-horizon-blend": 0.8,
        "atmosphere-blend": [
          "interpolate",
          ["linear"],
          ["zoom"],
          0, 1,
          6, 0.5,
          12, 0,
        ],
      },
    } as unknown as maplibregl.StyleSpecification;
  }

  private addSources(): void {
    const empty = { type: "FeatureCollection", features: [] } as const;
    for (const id of ["aoi", "positives", "negatives", "positive-matches", "results", "preview", "draft", "poly-draft"]) {
      this.map.addSource(id, { type: "geojson", data: empty as any });
    }
  }

  private addLayers(): void {
    // AOI bbox
    this.map.addLayer({
      id: "aoi-fill",
      type: "fill",
      source: "aoi",
      paint: {
        "fill-color": "#e5a853",
        "fill-opacity": 0.06,
      },
    });
    this.map.addLayer({
      id: "aoi-line",
      type: "line",
      source: "aoi",
      paint: {
        "line-color": "#e5a853",
        "line-width": 1.6,
        "line-dasharray": [2, 2],
      },
    });

    // Draft AOI while dragging
    this.map.addLayer({
      id: "draft-fill",
      type: "fill",
      source: "draft",
      paint: { "fill-color": "#e5a853", "fill-opacity": 0.08 },
    });
    this.map.addLayer({
      id: "draft-line",
      type: "line",
      source: "draft",
      paint: { "line-color": "#e5a853", "line-width": 1.6 },
    });

    // Live polygon draw preview
    this.map.addLayer({
      id: "poly-draft-line",
      type: "line",
      source: "poly-draft",
      paint: { "line-color": "#e5a853", "line-width": 1.6, "line-dasharray": [2, 2] },
    });
    this.map.addLayer({
      id: "poly-draft-vertices",
      type: "circle",
      source: "poly-draft",
      filter: ["==", "$type", "Point"],
      paint: { "circle-radius": 4, "circle-color": "#e5a853" },
    });

    // Ranked results
    this.map.addLayer({
      id: "results-fill",
      type: "fill",
      source: "results",
      paint: {
        "fill-color": ["coalesce", ["get", "color"], "#d0542c"],
        "fill-opacity": ["coalesce", ["get", "fillOpacity"], 0.18],
      },
    });
    this.map.addLayer({
      id: "results-line",
      type: "line",
      source: "results",
      paint: {
        "line-color": ["coalesce", ["get", "color"], "#d0542c"],
        "line-width": ["coalesce", ["get", "lineWidth"], 1.2],
        "line-opacity": 0.9,
      },
    });

    // Positive-match tiles (the exemplar patch underneath each point)
    this.map.addLayer({
      id: "positive-match-fill",
      type: "fill",
      source: "positive-matches",
      paint: { "fill-color": "#c74633", "fill-opacity": 0.16 },
    });
    this.map.addLayer({
      id: "positive-match-line",
      type: "line",
      source: "positive-matches",
      paint: { "line-color": "#c74633", "line-width": 1.6 },
    });

    // Preview (hover) bbox
    this.map.addLayer({
      id: "preview-line",
      type: "line",
      source: "preview",
      paint: {
        "line-color": "#ffffff",
        "line-width": 2,
        "line-dasharray": [3, 2],
      },
    });

    // Positive points
    this.map.addLayer({
      id: "positives-halo",
      type: "circle",
      source: "positives",
      paint: {
        "circle-radius": 11,
        "circle-color": "#c74633",
        "circle-opacity": 0.18,
      },
    });
    this.map.addLayer({
      id: "positives-dot",
      type: "circle",
      source: "positives",
      paint: {
        "circle-radius": 5,
        "circle-color": "#c74633",
        "circle-stroke-color": "#f3ecd8",
        "circle-stroke-width": 1.5,
      },
    });

    // Negative points (blue)
    this.map.addLayer({
      id: "negatives-halo",
      type: "circle",
      source: "negatives",
      paint: {
        "circle-radius": 11,
        "circle-color": "#3b82f6",
        "circle-opacity": 0.18,
      },
    });
    this.map.addLayer({
      id: "negatives-dot",
      type: "circle",
      source: "negatives",
      paint: {
        "circle-radius": 5,
        "circle-color": "#3b82f6",
        "circle-stroke-color": "#f3ecd8",
        "circle-stroke-width": 1.5,
      },
    });
  }

  private easeIntro(): void {
    this.map.easeTo({ center: [-95, 38], zoom: 2.2, duration: 2600, essential: true });
  }

  private whenReady(fn: () => void): void {
    if (this.styleReady) fn();
    else this.pendingRender.push(fn);
  }

  setAois(entries: Pick<AoiEntry, "bbox" | "polygon">[]): void {
    this.whenReady(() => {
      const src = this.map.getSource("aoi") as maplibregl.GeoJSONSource;
      src?.setData({
        type: "FeatureCollection",
        features: entries.map((e) =>
          e.polygon ? ringToFeature(e.polygon) : bboxToPolygon(e.bbox)
        ),
      });
    });
  }

  private setPolyDraft(vertices: maplibregl.LngLat[] | null): void {
    const src = this.map.getSource("poly-draft") as maplibregl.GeoJSONSource | undefined;
    if (!src) return;
    if (!vertices || vertices.length < 2) {
      src.setData({ type: "FeatureCollection", features: [] });
      return;
    }
    const coords = vertices.map((v) => [v.lng, v.lat] as [number, number]);
    const ring = coords.length >= 3 ? [...coords, coords[0]] : coords;
    src.setData({
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          geometry: { type: "LineString", coordinates: ring },
          properties: {},
        },
        ...coords.map((c) => ({
          type: "Feature" as const,
          geometry: { type: "Point" as const, coordinates: c },
          properties: {},
        })),
      ],
    });
  }

  setDraft(bbox: BBox | null): void {
    const src = this.map.getSource("draft") as maplibregl.GeoJSONSource | undefined;
    if (!src) return;
    src.setData({
      type: "FeatureCollection",
      features: bbox ? [bboxToPolygon(bbox)] : [],
    });
  }

  setPositives(points: PositivePoint[]): void {
    this.whenReady(() => {
      const src = this.map.getSource("positives") as maplibregl.GeoJSONSource;
      src?.setData({
        type: "FeatureCollection",
        features: points.map((p) => ({
          type: "Feature",
          geometry: { type: "Point", coordinates: [p.lng, p.lat] },
          properties: { id: p.id },
        })),
      });
    });
  }

  setNegatives(points: NegativePoint[]): void {
    this.whenReady(() => {
      const src = this.map.getSource("negatives") as maplibregl.GeoJSONSource;
      src?.setData({
        type: "FeatureCollection",
        features: points.map((p) => ({
          type: "Feature",
          geometry: { type: "Point", coordinates: [p.lng, p.lat] },
          properties: { id: p.id },
        })),
      });
    });
  }

  setPositiveMatches(matches: PositiveMatch[]): void {
    this.whenReady(() => {
      const src = this.map.getSource("positive-matches") as maplibregl.GeoJSONSource;
      src?.setData({
        type: "FeatureCollection",
        features: matches.map((m) => bboxToPolygon(m.candidate.bbox, { id: m.pointId })),
      });
    });
  }

  setResults(results: RankedRow[], topK: number, viewMode: ViewMode): void {
    this.whenReady(() => {
      const src = this.map.getSource("results") as maplibregl.GeoJSONSource;
      if (!src) return;

      if (!results.length) {
        if (this._lastResultsKey) {
          src.setData({ type: "FeatureCollection", features: [] });
          this._lastResultsKey = "";
        }
        this.map.setFilter("results-fill", null);
        this.map.setFilter("results-line", null);
        return;
      }

      const n = results.length;
      const dataKey = results.map((r) => r.chips_id).join("\0");
      if (dataKey !== this._lastResultsKey) {
        const features = results.map((r, i) => {
          const t = n > 1 ? i / (n - 1) : 0;
          return bboxToPolygon(r.bbox, {
            chipsId: r.chips_id,
            score: r.score,
            rank: i,
            heatColor: interpolatePlasma(t),
            heatFillOpacity: 0.28 - t * 0.14,
          });
        });
        src.setData({ type: "FeatureCollection", features });
        this._lastResultsKey = dataKey;
      }

      this._applyViewStyle(viewMode, topK, n);
    });
  }

  private _applyViewStyle(viewMode: ViewMode, topK: number, total: number): void {
    const rankFilter = viewMode === "topk" && total > 0
      ? (["<", ["get", "rank"], topK] as any)
      : null;
    this.map.setFilter("results-fill", rankFilter);
    this.map.setFilter("results-line", rankFilter);

    if (viewMode !== "topk") {
      this.map.setPaintProperty("results-fill", "fill-color", ["coalesce", ["get", "heatColor"], "#d0542c"]);
      this.map.setPaintProperty("results-fill", "fill-opacity", ["coalesce", ["get", "heatFillOpacity"], 0.18]);
      this.map.setPaintProperty("results-line", "line-color", ["coalesce", ["get", "heatColor"], "#d0542c"]);
      this.map.setPaintProperty("results-line", "line-width", 0.6);
      this.map.setPaintProperty("results-line", "line-opacity", 0.9);
    } else {
      this.map.setPaintProperty("results-fill", "fill-color", "#d0542c");
      this.map.setPaintProperty("results-fill", "fill-opacity",
        total > 1
          ? (["interpolate", ["linear"], ["get", "rank"], 0, 0.18, Math.min(topK - 1, total - 1), 0.08] as any)
          : 0.18,
      );
      this.map.setPaintProperty("results-line", "line-color", "#d0542c");
      this.map.setPaintProperty("results-line", "line-width", 1.4);
      this.map.setPaintProperty("results-line", "line-opacity", 0.9);
    }
  }

  setPreview(result: RankedRow | null): void {
    this.whenReady(() => {
      const src = this.map.getSource("preview") as maplibregl.GeoJSONSource;
      src?.setData({
        type: "FeatureCollection",
        features: result ? [bboxToPolygon(result.bbox)] : [],
      });
    });
  }

  flyToBBox(bbox: BBox, opts: { zoom?: number } = {}): void {
    const c = centroid(bbox);
    this.map.flyTo({
      center: [c.lng, c.lat],
      zoom: opts.zoom ?? Math.max(this.map.getZoom(), 10),
      speed: 1.2,
      curve: 1.4,
      essential: true,
    });
  }

  fitBounds(bbox: BBox, opts: { padding?: number; maxZoom?: number } = {}): void {
    this.map.fitBounds(
      [
        [bbox.west, bbox.south],
        [bbox.east, bbox.north],
      ],
      {
        padding: opts.padding ?? 80,
        maxZoom: opts.maxZoom ?? 12,
        duration: 1800,
        curve: 1.4,
        essential: true,
      },
    );
  }

  armDraw(on: boolean, mode: DrawMode = "rect"): void {
    this.draw.armed = on;
    this.draw.mode = mode;
    const c = this.map.getCanvas();
    c.style.cursor = on ? "crosshair" : "";
  }

  isArmed(): boolean {
    return this.draw.armed;
  }

  getDrawMode(): DrawMode {
    return this.draw.mode;
  }

  cancelDraft(): void {
    this.draw.startLngLat = null;
    this.draw.startPoint = null;
    this.draw.moved = false;
    this.draw.polyVertices = [];
    this.draw.polyActive = false;
    this.removeDomBox();
    this.setDraft(null);
    this.setPolyDraft(null);
    this.map.dragPan.enable();
  }

  private wireDrawing(): void {
    const canvas = () => this.map.getCanvas();

    // ── Shared draw-start / draw-move / draw-end for both mouse + touch ──────

    const drawStart = (lngLat: maplibregl.LngLat, point: { x: number; y: number }, ev: Event) => {
      ev.preventDefault();
      this.map.dragPan.disable();
      this.draw.startLngLat = lngLat;
      this.draw.startPoint = point;
      this.draw.moved = false;
      this.ensureDomBox(point.x, point.y);
    };

    const drawMove = (lngLat: maplibregl.LngLat, point: { x: number; y: number }) => {
      if (this.draw.mode === "rect" && this.draw.startLngLat && this.draw.startPoint) {
        this.draw.moved = true;
        this.updateDomBox(point.x, point.y);
        this.setDraft(bboxFromLngLats(this.draw.startLngLat, lngLat));
      } else if (this.draw.mode === "polygon" && this.draw.polyActive) {
        this.setPolyDraft([...this.draw.polyVertices, lngLat]);
      }
    };

    const drawEnd = (lngLat: maplibregl.LngLat) => {
      if (this.draw.mode !== "rect" || !this.draw.startLngLat) return;
      const start = this.draw.startLngLat;
      const moved = this.draw.moved;
      this.draw.startLngLat = null;
      this.draw.startPoint = null;
      this.draw.moved = false;
      this.removeDomBox();
      this.setDraft(null);
      this.map.dragPan.enable();
      this.draw.armed = false;
      canvas().style.cursor = "";
      if (!moved) return;
      const bbox = bboxFromLngLats(start, lngLat);
      this.cb.onDrawComplete({ bbox });
    };

    // ── Rectangle mode (mouse) ────────────────────────────────────────────────
    this.map.on("mousedown", (e) => {
      if (!(e.originalEvent.shiftKey || this.draw.armed) || this.draw.mode !== "rect") return;
      drawStart(e.lngLat, e.point, e.originalEvent);
    });
    this.map.on("mousemove", (e) => drawMove(e.lngLat, e.point));
    this.map.on("mouseup", (e) => drawEnd(e.lngLat));

    // ── Rectangle mode (touch) ────────────────────────────────────────────────
    this.map.on("touchstart", (e) => {
      if (!this.draw.armed || this.draw.mode !== "rect") return;
      drawStart(e.lngLat, e.point, e.originalEvent);
    });
    this.map.on("touchmove", (e) => {
      if (!this.draw.startLngLat) return;
      e.originalEvent.preventDefault();
      drawMove(e.lngLat, e.point);
    });
    this.map.on("touchend", (e) => {
      if (!this.draw.startLngLat) return;
      drawEnd(e.lngLat);
    });

    // ── Polygon mode ─────────────────────────────────────────────────────────
    this.map.on("click", (e) => {
      if (!this.draw.armed || this.draw.mode !== "polygon") return;
      if (e.originalEvent.detail === 2) return; // skip second click of a dblclick
      this.draw.polyActive = true;
      this.draw.polyVertices.push(e.lngLat);
      this.setPolyDraft([...this.draw.polyVertices]);
    });

    this.map.on("dblclick", (e) => {
      if (!this.draw.armed || this.draw.mode !== "polygon") return;
      if (this.draw.polyVertices.length < 3) return;
      e.preventDefault();
      const verts = this.draw.polyVertices;
      const ring: [number, number][] = [
        ...verts.map((v) => [v.lng, v.lat] as [number, number]),
        [verts[0].lng, verts[0].lat],
      ];
      const bbox = ringToBBox(ring);
      this.draw.polyVertices = [];
      this.draw.polyActive = false;
      this.draw.armed = false;
      canvas().style.cursor = "";
      this.setPolyDraft(null);
      this.cb.onDrawComplete({ bbox, polygon: ring });
    });
  }

  private wireClicks(): void {
    // Right-click → negative exemplar
    this.map.on("contextmenu", (e) => {
      e.originalEvent.preventDefault();
      if (this.draw.startLngLat) return;
      const { lat, lng } = e.lngLat;
      this.cb.onNegativeClick(lat, lng);
    });

    // Long-press (touch) → negative exemplar (mobile equivalent of right-click)
    let longPressTimer: ReturnType<typeof setTimeout> | null = null;
    let longPressLngLat: { lat: number; lng: number } | null = null;
    this.map.on("touchstart", (e) => {
      if (this.draw.armed) return;
      if (e.originalEvent.touches.length !== 1) return;
      longPressLngLat = e.lngLat;
      longPressTimer = setTimeout(() => {
        if (longPressLngLat) {
          this.cb.onNegativeClick(longPressLngLat.lat, longPressLngLat.lng);
        }
        longPressTimer = null;
        longPressLngLat = null;
      }, 500);
    });
    const cancelLongPress = () => {
      if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = null; }
      longPressLngLat = null;
    };
    this.map.on("touchmove", cancelLongPress);
    this.map.on("touchend", cancelLongPress);
    this.map.on("touchcancel", cancelLongPress);

    this.map.on("click", (e) => {
      if (this.draw.armed && this.draw.mode === "polygon") return;
      if (this.draw.startLngLat) return;
      // Shift+click → negative exemplar
      if (e.originalEvent.shiftKey && !this.draw.armed) {
        const { lat, lng } = e.lngLat;
        this.cb.onNegativeClick(lat, lng);
        return;
      }
      // Click on a result tile -> treat as picking an exemplar
      const hits = this.map.queryRenderedFeatures(e.point, {
        layers: ["results-fill"],
      });
      if (hits.length) {
        const props = hits[0].properties as { chipsId?: string };
        const results = this.cb.getResults();
        const row = results.find((r) => r.chips_id === props.chipsId);
        if (row) {
          this.cb.onResultPick(row);
          return;
        }
      }
      const { lat, lng } = e.lngLat;
      this.cb.onAoiClick(lat, lng);
    });

    this.map.on("mousemove", "results-fill", (e) => {
      const f = e.features?.[0];
      if (!f) return;
      const props = f.properties as { chipsId?: string };
      const row = this.cb.getResults().find((r) => r.chips_id === props.chipsId);
      if (row) this.cb.onResultHover(row);
      this.map.getCanvas().style.cursor = this.draw.armed ? "crosshair" : "pointer";
    });
    this.map.on("mouseleave", "results-fill", () => {
      this.cb.onResultHover(null);
      if (!this.draw.armed) this.map.getCanvas().style.cursor = "";
    });
  }

  private ensureDomBox(x: number, y: number): void {
    if (this.draw.box) return;
    const box = document.createElement("div");
    box.className = "draw-box";
    box.style.left = `${x}px`;
    box.style.top = `${y}px`;
    box.style.width = "0px";
    box.style.height = "0px";
    this.map.getCanvasContainer().appendChild(box);
    this.draw.box = box;
  }

  private updateDomBox(x: number, y: number): void {
    if (!this.draw.box || !this.draw.startPoint) return;
    const sx = this.draw.startPoint.x;
    const sy = this.draw.startPoint.y;
    const left = Math.min(sx, x);
    const top = Math.min(sy, y);
    const w = Math.abs(x - sx);
    const h = Math.abs(y - sy);
    this.draw.box.style.left = `${left}px`;
    this.draw.box.style.top = `${top}px`;
    this.draw.box.style.width = `${w}px`;
    this.draw.box.style.height = `${h}px`;
  }

  private removeDomBox(): void {
    this.draw.box?.remove();
    this.draw.box = null;
  }
}

function bboxFromLngLats(a: maplibregl.LngLat, b: maplibregl.LngLat): BBox {
  return {
    west: Math.min(a.lng, b.lng),
    east: Math.max(a.lng, b.lng),
    south: Math.min(a.lat, b.lat),
    north: Math.max(a.lat, b.lat),
  };
}

function ringToFeature(ring: [number, number][], properties: Record<string, unknown> = {}): GeoJSON.Feature {
  return {
    type: "Feature",
    properties,
    geometry: { type: "Polygon", coordinates: [ring] },
  };
}

function ringToBBox(ring: [number, number][]): BBox {
  const lngs = ring.map((c) => c[0]);
  const lats = ring.map((c) => c[1]);
  return {
    west: Math.min(...lngs),
    east: Math.max(...lngs),
    south: Math.min(...lats),
    north: Math.max(...lats),
  };
}

function bboxToPolygon(bbox: BBox, properties: Record<string, unknown> = {}): GeoJSON.Feature {
  return {
    type: "Feature",
    properties,
    geometry: {
      type: "Polygon",
      coordinates: [
        [
          [bbox.west, bbox.south],
          [bbox.east, bbox.south],
          [bbox.east, bbox.north],
          [bbox.west, bbox.north],
          [bbox.west, bbox.south],
        ],
      ],
    },
  };
}
