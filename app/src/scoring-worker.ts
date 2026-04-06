type InitMessage = {
  type: "init";
  embeddings: Uint8Array[];
  centroids: Float64Array;
};

type ScoreMessage = {
  type: "score";
  requestId: number;
  exemplars: Uint8Array[];
  negatives: Uint8Array[];
  excludeIndices: number[];
};

type OutlierMessage = {
  type: "outlier";
  requestId: number;
  sampleSize: number;
};

type SurpriseMessage = {
  type: "surprise";
  requestId: number;
  k: number;
};

type GradientMessage = {
  type: "gradient";
  requestId: number;
  scores: Float64Array;
  k: number;
};

type WorkerMessage = InitMessage | ScoreMessage | OutlierMessage | SurpriseMessage | GradientMessage;

type ScoredResult = {
  index: number;
  score: number;
};

const popcountTable = new Uint8Array(256);
for (let i = 0; i < 256; i += 1) {
  let value = i;
  let count = 0;
  while (value > 0) {
    count += value & 1;
    value >>= 1;
  }
  popcountTable[i] = count;
}

let candidateEmbeddings: Uint8Array[] = [];
let candidateCentroids: Float64Array = new Float64Array(0);

function hammingDistance(a: Uint8Array, b: Uint8Array): number {
  let total = 0;
  const length = Math.min(a.length, b.length);
  for (let i = 0; i < length; i += 1) {
    total += popcountTable[a[i] ^ b[i]];
  }
  return total;
}

function scoreCandidates(
  exemplars: Uint8Array[],
  negatives: Uint8Array[],
  excludeIndices: Set<number>,
): ScoredResult[] {
  const results: ScoredResult[] = [];
  const hasNeg = negatives.length > 0;
  for (let index = 0; index < candidateEmbeddings.length; index += 1) {
    if (excludeIndices.has(index)) continue;
    const candidate = candidateEmbeddings[index];
    let posScore = 0;
    for (const exemplar of exemplars) {
      posScore += hammingDistance(candidate, exemplar);
    }
    posScore /= exemplars.length;

    if (hasNeg) {
      let negScore = 0;
      for (const neg of negatives) {
        negScore += hammingDistance(candidate, neg);
      }
      negScore /= negatives.length;
      results.push({ index, score: posScore - negScore });
    } else {
      results.push({ index, score: posScore });
    }
  }

  results.sort((a, b) => a.score - b.score || a.index - b.index);
  return results;
}

function computeOutlierScores(sampleSize: number): ScoredResult[] {
  const n = candidateEmbeddings.length;
  if (n === 0) return [];

  const m = Math.min(sampleSize, n);
  const step = Math.max(1, Math.floor(n / m));
  const refs: Uint8Array[] = [];
  for (let i = 0; i < n && refs.length < m; i += step) {
    refs.push(candidateEmbeddings[i]);
  }

  const results: ScoredResult[] = [];
  for (let i = 0; i < n; i += 1) {
    const emb = candidateEmbeddings[i];
    let total = 0;
    for (const ref of refs) {
      total += hammingDistance(emb, ref);
    }
    results.push({ index: i, score: total / refs.length });
  }

  results.sort((a, b) => b.score - a.score || a.index - b.index);
  return results;
}

function centroidDistSq(i: number, j: number): number {
  const latI = candidateCentroids[i * 2];
  const lngI = candidateCentroids[i * 2 + 1];
  const latJ = candidateCentroids[j * 2];
  const lngJ = candidateCentroids[j * 2 + 1];
  const dLat = latI - latJ;
  const dLng = lngI - lngJ;
  return dLat * dLat + dLng * dLng;
}

function findKNearestGeo(idx: number, k: number): number[] {
  const n = candidateEmbeddings.length;
  const neighbors: { i: number; d: number }[] = [];
  for (let j = 0; j < n; j += 1) {
    if (j === idx) continue;
    const d = centroidDistSq(idx, j);
    if (neighbors.length < k) {
      neighbors.push({ i: j, d });
      if (neighbors.length === k) neighbors.sort((a, b) => b.d - a.d);
    } else if (d < neighbors[0].d) {
      neighbors[0] = { i: j, d };
      neighbors.sort((a, b) => b.d - a.d);
    }
  }
  return neighbors.map((n) => n.i);
}

function computeSurpriseScores(k: number): ScoredResult[] {
  const n = candidateEmbeddings.length;
  if (n === 0) return [];

  const results: ScoredResult[] = [];
  for (let i = 0; i < n; i += 1) {
    const neighbors = findKNearestGeo(i, k);
    if (!neighbors.length) {
      results.push({ index: i, score: 0 });
      continue;
    }
    let total = 0;
    for (const j of neighbors) {
      total += hammingDistance(candidateEmbeddings[i], candidateEmbeddings[j]);
    }
    results.push({ index: i, score: total / neighbors.length });
  }

  results.sort((a, b) => b.score - a.score || a.index - b.index);
  return results;
}

function computeGradientScores(scores: Float64Array, k: number): ScoredResult[] {
  const n = candidateEmbeddings.length;
  if (n === 0) return [];

  const results: ScoredResult[] = [];
  for (let i = 0; i < n; i += 1) {
    const neighbors = findKNearestGeo(i, k);
    if (!neighbors.length) {
      results.push({ index: i, score: 0 });
      continue;
    }
    let total = 0;
    for (const j of neighbors) {
      total += Math.abs(scores[i] - scores[j]);
    }
    results.push({ index: i, score: total / neighbors.length });
  }

  results.sort((a, b) => b.score - a.score || a.index - b.index);
  return results;
}

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  if (event.data.type === "init") {
    candidateEmbeddings = event.data.embeddings.map((embedding) => new Uint8Array(embedding));
    candidateCentroids = new Float64Array(event.data.centroids);
    return;
  }

  if (event.data.type === "outlier") {
    const results = computeOutlierScores(event.data.sampleSize);
    self.postMessage({
      type: "outlier-result",
      requestId: event.data.requestId,
      results,
    });
    return;
  }

  if (event.data.type === "surprise") {
    const results = computeSurpriseScores(event.data.k);
    self.postMessage({
      type: "surprise-result",
      requestId: event.data.requestId,
      results,
    });
    return;
  }

  if (event.data.type === "gradient") {
    const results = computeGradientScores(
      new Float64Array(event.data.scores),
      event.data.k,
    );
    self.postMessage({
      type: "gradient-result",
      requestId: event.data.requestId,
      results,
    });
    return;
  }

  const exemplars = event.data.exemplars.map((embedding) => new Uint8Array(embedding));
  const negatives = event.data.negatives.map((embedding) => new Uint8Array(embedding));
  const results = scoreCandidates(exemplars, negatives, new Set(event.data.excludeIndices));
  self.postMessage({
    type: "score-result",
    requestId: event.data.requestId,
    results,
  });
};
