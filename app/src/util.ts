import type { BBox } from "./types";

export function centroid(box: BBox): { lat: number; lng: number } {
  return {
    lat: (box.south + box.north) / 2,
    lng: (box.west + box.east) / 2,
  };
}

export function containsPoint(box: BBox, lat: number, lng: number): boolean {
  return lng >= box.west && lng <= box.east && lat >= box.south && lat <= box.north;
}

export function distanceSquared(aLat: number, aLng: number, bLat: number, bLng: number): number {
  const dLat = aLat - bLat;
  const dLng = aLng - bLng;
  return dLat * dLat + dLng * dLng;
}

export function normalizeBBox(value: unknown): BBox {
  const box = value as { xmin: number; ymin: number; xmax: number; ymax: number };
  return { west: box.xmin, south: box.ymin, east: box.xmax, north: box.ymax };
}

export function normalizeEmbedding(value: unknown): Uint8Array {
  if (value instanceof Uint8Array) return value;
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength));
  }
  if (Array.isArray(value)) return Uint8Array.from(value.map((v) => Number(v)));
  if (typeof value === "object" && value !== null) {
    const c = value as {
      toArray?: () => unknown;
      values?: () => Iterable<unknown>;
      length?: number;
      [index: number]: unknown;
    };
    if (typeof c.toArray === "function") return normalizeEmbedding(c.toArray());
    if (typeof c.values === "function") return normalizeEmbedding(Array.from(c.values()));
    if (typeof c.length === "number") {
      return Uint8Array.from(Array.from({ length: c.length }, (_, i) => Number(c[i])));
    }
  }
  throw new TypeError("Unexpected embedding value");
}

export function formatLatLng(lat: number, lng: number): string {
  const ns = lat >= 0 ? "N" : "S";
  const ew = lng >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(3)}°${ns}  ${Math.abs(lng).toFixed(3)}°${ew}`;
}

export function interpolatePlasma(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const stops = [
    { t: 0.0, r: 240, g: 249, b: 33 },
    { t: 0.25, r: 248, g: 149, b: 64 },
    { t: 0.5, r: 204, g: 71, b: 120 },
    { t: 0.75, r: 126, g: 3, b: 167 },
    { t: 1.0, r: 13, g: 8, b: 135 },
  ];
  for (let i = 0; i < stops.length - 1; i += 1) {
    const s = stops[i];
    const e = stops[i + 1];
    if (clamped >= s.t && clamped <= e.t) {
      const lt = (clamped - s.t) / (e.t - s.t);
      const mix = (a: number, b: number) => Math.round(a + (b - a) * lt);
      return `rgb(${mix(s.r, e.r)} ${mix(s.g, e.g)} ${mix(s.b, e.b)})`;
    }
  }
  const l = stops[stops.length - 1];
  return `rgb(${l.r} ${l.g} ${l.b})`;
}
