ey Isaac! nice work, there's not much out there on storage optimization for geo embeddings so this fills a real gap. Some few comments!

1. you're storing the quantized embeddings as a blob column right? have you tried BYTE_STREAM_SPLIT encoding in parquet before compression? it transposes the byte planes across rows so identical byte positions from all embeddings in a row group end up together. within a geohash partition where embeddings are similar, those byte streams become super compressible. it's free compression on top of what you already have, no schema change needed. would be cool to see how much extra compression you get from that alone.

2. since per-partition calibration drifts across geohash cells (slide 9), wouldn't it make sense to compute ID per-geohash instead of one global estimate? could be interesting as future work to see how much it varies across partitions.

3. would be interesting to compare OPQ vs turboquant. in theory OPQ should do better at 2-3 bits since it learns the rotation that minimizes quantization error instead of a random ones. (I did not read the turboquant paper yet!)