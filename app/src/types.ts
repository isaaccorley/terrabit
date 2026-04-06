export type BBox = {
  west: number;
  south: number;
  east: number;
  north: number;
};

export type ManifestRow = {
  path: string;
  rows: number;
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  year?: string;
};

export type PositivePoint = {
  id: number;
  lat: number;
  lng: number;
};

export type CandidateRow = {
  chips_id: string;
  bbox: BBox;
  embedding: Uint8Array;
  shard_path: string;
};

export type PositiveMatch = {
  pointId: number;
  candidate: CandidateRow;
};

export type RankedRow = CandidateRow & {
  score: number;
};

export type ViewMode = "topk" | "heatmap" | "outlier" | "threshold";
