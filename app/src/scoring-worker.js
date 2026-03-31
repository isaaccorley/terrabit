"use strict";
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
let candidateEmbeddings = [];
function hammingDistance(a, b) {
    let total = 0;
    const length = Math.min(a.length, b.length);
    for (let i = 0; i < length; i += 1) {
        total += popcountTable[a[i] ^ b[i]];
    }
    return total;
}
function scoreCandidates(exemplars, excludeIndices) {
    const results = [];
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
self.onmessage = (event) => {
    if (event.data.type === "init") {
        candidateEmbeddings = event.data.embeddings.map((embedding) => new Uint8Array(embedding));
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
