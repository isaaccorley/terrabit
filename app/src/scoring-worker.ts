type InitMessage = {
  type: "init";
  embeddings: Uint8Array[];
};

type ScoreMessage = {
  type: "score";
  requestId: number;
  exemplars: Uint8Array[];
  excludeIndices: number[];
};

type OutlierMessage = {
  type: "outlier";
  requestId: number;
  sampleSize: number;
};

type WorkerMessage = InitMessage | ScoreMessage | OutlierMessage;

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

function hammingDistance(a: Uint8Array, b: Uint8Array): number {
  let total = 0;
  const length = Math.min(a.length, b.length);
  for (let i = 0; i < length; i += 1) {
    total += popcountTable[a[i] ^ b[i]];
  }
  return total;
}

function scoreCandidates(exemplars: Uint8Array[], excludeIndices: Set<number>): ScoredResult[] {
  const results: ScoredResult[] = [];
  for (let index = 0; index < candidateEmbeddings.length; index += 1) {
    if (excludeIndices.has(index)) {
      continue;
    }
    const candidate = candidateEmbeddings[index];
    let score = 0;
    for (const exemplar of exemplars) {
      score += hammingDistance(candidate, exemplar);
    }
    results.push({ index, score: score / exemplars.length });
  }

  results.sort((a, b) => a.score - b.score || a.index - b.index);
  return results;
}

function computeOutlierScores(sampleSize: number): ScoredResult[] {
  const n = candidateEmbeddings.length;
  if (n === 0) return [];

  // Deterministic sampling: evenly spaced reference embeddings
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
    // Higher mean distance → more unique / outlier-like
    results.push({ index: i, score: total / refs.length });
  }

  // Descending: most unique first
  results.sort((a, b) => b.score - a.score || a.index - b.index);
  return results;
}

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  if (event.data.type === "init") {
    candidateEmbeddings = event.data.embeddings.map((embedding) => new Uint8Array(embedding));
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

  const exemplars = event.data.exemplars.map((embedding) => new Uint8Array(embedding));
  const results = scoreCandidates(exemplars, new Set(event.data.excludeIndices));
  self.postMessage({
    type: "score-result",
    requestId: event.data.requestId,
    results,
  });
};
