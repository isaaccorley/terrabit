import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

import type {
  BBox,
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
  onDrawComplete: (bbox: BBox) => void;
  onAoiClick: (lat: number, lng: number) => void;
  onResultHover: (result: RankedRow | null) => void;
  onResultPick: (result: RankedRow) => void;
  getBBox: () => BBox | null;
  getResults: () => RankedRow[];
  getTopK: () => number;
};

type DrawState = {
  startLngLat: maplibregl.LngLat | null;
  startPoint: { x: number; y: number } | null;
  moved: boolean;
  armed: boolean;
  box: HTMLDivElement | null;
};

export class GlobeMap {
  readonly map: maplibregl.Map;
  private cb: MapCallbacks;
  private draw: DrawState = {
    startLngLat: null,
    startPoint: null,
    moved: false,
    armed: false,
    box: null,
  };
  private styleReady = false;
  private pendingRender: (() => void)[] = [];

  constructor(container: HTMLElement, cb: MapCallbacks) {
    this.cb = cb;
    this.map = new maplibregl.Map({
      container,
      style: this.buildStyle(),
      center: [8, 22],
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
    for (const id of ["aoi", "positives", "positive-matches", "results", "contour", "preview", "draft"]) {
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

    // Contour (MapLibre native heatmap layer — smooth gaussian interpolation)
    this.map.addLayer({
      id: "contour-heat",
      type: "heatmap",
      source: "contour",
      layout: { visibility: "none" },
      paint: {
        "heatmap-weight": ["coalesce", ["get", "weight"], 0.5],
        "heatmap-intensity": [
          "interpolate", ["linear"], ["zoom"],
          4, 0.6,
          8, 1.2,
          12, 2,
        ],
        "heatmap-radius": [
          "interpolate", ["linear"], ["zoom"],
          4, 6,
          8, 18,
          12, 36,
        ],
        "heatmap-opacity": 0.7,
        "heatmap-color": [
          "interpolate", ["linear"], ["heatmap-density"],
          0, "rgba(13, 8, 135, 0)",
          0.15, "rgba(126, 3, 167, 0.45)",
          0.35, "rgba(204, 71, 120, 0.65)",
          0.55, "rgba(248, 149, 64, 0.8)",
          0.8, "rgba(240, 249, 33, 0.9)",
          1, "rgba(240, 249, 33, 1)",
        ],
      },
    } as any);

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
  }

  private easeIntro(): void {
    this.map.easeTo({ center: [8, 22], zoom: 2.2, duration: 2600, essential: true });
  }

  private whenReady(fn: () => void): void {
    if (this.styleReady) fn();
    else this.pendingRender.push(fn);
  }

  setAoi(bbox: BBox | null): void {
    this.whenReady(() => {
      const src = this.map.getSource("aoi") as maplibregl.GeoJSONSource;
      src?.setData({
        type: "FeatureCollection",
        features: bbox ? [bboxToPolygon(bbox)] : [],
      });
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
      const isContour = viewMode === "contour";

      // Toggle contour heatmap layer
      if (this.map.getLayer("contour-heat")) {
        this.map.setLayoutProperty("contour-heat", "visibility", isContour ? "visible" : "none");
      }

      if (isContour) {
        // Set contour point source (centroids with weights)
        const n = results.length;
        const features: GeoJSON.Feature[] = results.map((r, i) => {
          const c = centroid(r.bbox);
          const weight = n > 1 ? 1 - i / (n - 1) : 1;
          return {
            type: "Feature",
            geometry: { type: "Point", coordinates: [c.lng, c.lat] },
            properties: { weight },
          };
        });
        (this.map.getSource("contour") as maplibregl.GeoJSONSource)?.setData({
          type: "FeatureCollection",
          features,
        });
        // Clear tile layers
        (this.map.getSource("results") as maplibregl.GeoJSONSource)?.setData({
          type: "FeatureCollection",
          features: [],
        });
        return;
      }

      // Clear contour data
      (this.map.getSource("contour") as maplibregl.GeoJSONSource)?.setData({
        type: "FeatureCollection",
        features: [],
      });

      // Tile-based views
      const useColor = viewMode !== "topk";
      const list = viewMode === "topk" ? results.slice(0, topK) : results;

      if (!list.length) {
        (this.map.getSource("results") as maplibregl.GeoJSONSource)?.setData({
          type: "FeatureCollection",
          features: [],
        });
        return;
      }
      const n = list.length;
      const features = list.map((r, i) => {
        const t = n > 1 ? i / (n - 1) : 0;
        const color = useColor ? interpolatePlasma(t) : "#d0542c";
        const fillOpacity = useColor ? 0.38 - t * 0.22 : 0.18 - t * 0.1;
        const lineWidth = useColor ? 0.6 : 1.4;
        return bboxToPolygon(r.bbox, {
          chipsId: r.chips_id,
          score: r.score,
          color,
          fillOpacity,
          lineWidth,
        });
      });
      (this.map.getSource("results") as maplibregl.GeoJSONSource)?.setData({
        type: "FeatureCollection",
        features,
      });
    });
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

  armDraw(on: boolean): void {
    this.draw.armed = on;
    const c = this.map.getCanvas();
    c.style.cursor = on ? "crosshair" : "";
  }

  isArmed(): boolean {
    return this.draw.armed;
  }

  cancelDraft(): void {
    this.draw.startLngLat = null;
    this.draw.startPoint = null;
    this.draw.moved = false;
    this.removeDomBox();
    this.setDraft(null);
    this.map.dragPan.enable();
  }

  private wireDrawing(): void {
    const canvas = () => this.map.getCanvas();

    this.map.on("mousedown", (e) => {
      const ev = e.originalEvent;
      if (!ev.shiftKey && !this.draw.armed) return;
      ev.preventDefault();
      this.map.dragPan.disable();
      this.draw.startLngLat = e.lngLat;
      this.draw.startPoint = { x: e.point.x, y: e.point.y };
      this.draw.moved = false;
      this.ensureDomBox(e.point.x, e.point.y);
    });

    this.map.on("mousemove", (e) => {
      if (!this.draw.startLngLat || !this.draw.startPoint) return;
      this.draw.moved = true;
      this.updateDomBox(e.point.x, e.point.y);
      const bbox = bboxFromLngLats(this.draw.startLngLat, e.lngLat);
      this.setDraft(bbox);
    });

    this.map.on("mouseup", (e) => {
      if (!this.draw.startLngLat) return;
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
      const bbox = bboxFromLngLats(start, e.lngLat);
      this.cb.onDrawComplete(bbox);
    });
  }

  private wireClicks(): void {
    this.map.on("click", (e) => {
      if (this.draw.startLngLat) return;
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
      const bbox = this.cb.getBBox();
      if (!bbox) return;
      const { lat, lng } = e.lngLat;
      if (lng < bbox.west || lng > bbox.east || lat < bbox.south || lat > bbox.north) return;
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
