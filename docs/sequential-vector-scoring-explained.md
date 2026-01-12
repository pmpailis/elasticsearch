# Sequential Vector Scoring - Deep Dive

## Overview

Sequential Vector Scoring is the process of reading vectors from disk and scoring them one posting list at a time. This document explains **exactly where vectors are read from** and **how they're processed**.

---

## Part 1: Where Vectors Are Stored on Disk

### File Structure

Vectors are stored in the **`.clivf` (cluster/posting list) file**. Each posting list follows this layout:

```
┌─────────────────────────────────────────────────────────────┐
│ POSTING LIST FOR CENTROID C                                 │
├─────────────────────────────────────────────────────────────┤
│ [HEADER]                                                     │
│   rawCentroid[dimension]         float32 × dimension        │
│   centroidDp                     float32 (1 value)          │
│   vectorCount                    VInt                       │
│   docIdEncoding                  byte (1 value)             │
├─────────────────────────────────────────────────────────────┤
│ [BULK BLOCK 0] (16 vectors)                                 │
│   ┌─────────────────────────────────────────────────────────┤
│   │ DOC IDS (FIRST!)                                        │
│   │   [delta-encoded doc IDs for 16 vectors]               │
│   │   Size: 16-64 bytes (depends on encoding)              │
│   ├─────────────────────────────────────────────────────────┤
│   │ QUANTIZED VECTORS (SECOND!)                             │
│   │   vector[0]   byte[quantizedByteSize]                  │
│   │   vector[1]   byte[quantizedByteSize]                  │
│   │   ...                                                   │
│   │   vector[15]  byte[quantizedByteSize]                  │
│   ├─────────────────────────────────────────────────────────┤
│   │ CORRECTIONS (THIRD!)                                    │
│   │   lowerIntervals[16]        float32 × 16               │
│   │   upperIntervals[16]        float32 × 16               │
│   │   componentSums[16]         short × 16                 │
│   │   additionalCorrections[16] float32 × 16               │
│   └─────────────────────────────────────────────────────────┤
│ [BULK BLOCK 1] (16 vectors)                                 │
│   DOC IDS → VECTORS → CORRECTIONS                           │
│ [BULK BLOCK N-1] (16 vectors)                               │
│   DOC IDS → VECTORS → CORRECTIONS                           │
│ [TAIL BLOCK] (< 16 vectors, individual scoring)             │
│   For each vector:                                          │
│     docId + vector + corrections                            │
└─────────────────────────────────────────────────────────────┘
```

**Key Points:**
- **Doc IDs come FIRST** in each block (allows pre-filtering before reading vectors)
- Vectors are **quantized** (1-7 bits per dimension, packed into bytes)
- Stored in **blocks of 16** for SIMD efficiency
- **Corrections stored LAST** after all quantized vectors in the block
- Doc IDs stored **delta-encoded** for compression
- **Reading is sequential forward** - never seeks backward

---

## Part 2: The Sequential Reading Process

### Entry Point: `visit()` Method

**File:** `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsReader.java`
**Lines:** 721-793

```java
@Override
public int visit(KnnCollector knnCollector) throws IOException {
    // 1. Seek to posting list start
    indexInput.seek(slicePos);

    int scoredDocs = 0;
    int limit = vectors - BULK_SIZE + 1;  // BULK_SIZE = 16
    int i = 0;

    // 2. Process full batches (16 vectors each)
    for (; i < limit; i += BULK_SIZE) {
        // STEP A: Read doc IDs
        readDocIds(BULK_SIZE);

        // STEP B: Check how many pass filter (pre-filtering)
        final int docsToBulkScore = acceptDocs == null
            ? BULK_SIZE
            : docToBulkScore(docIdsScratch, acceptDocs);

        if (docsToBulkScore == 0) {
            // All filtered out - skip reading vectors
            indexInput.skipBytes(quantizedByteLength * BULK_SIZE);
            continue;
        }

        // STEP C: Quantize query (once per posting list)
        quantizeQueryIfNecessary();

        // STEP D: Choose scoring strategy
        final float maxScore;
        if (docsToBulkScore < BULK_SIZE / 2) {
            // < 50% pass filter: score individually
            maxScore = scoreIndividually();
        } else {
            // >= 50% pass filter: bulk SIMD scoring
            maxScore = osqVectorsScorer.scoreBulk(
                quantizedQueryScratch,
                queryCorrections.lowerInterval(),
                queryCorrections.upperInterval(),
                queryCorrections.quantizedComponentSum(),
                queryCorrections.additionalCorrection(),
                fieldInfo.getVectorSimilarityFunction(),
                centroidDp,
                scores
            );
        }

        // STEP E: Collect results above threshold
        if (knnCollector.minCompetitiveSimilarity() < maxScore) {
            collectBulk(knnCollector, scores);
        }

        scoredDocs += docsToBulkScore;
    }

    // 3. Process tail (< 16 vectors) individually
    if (i < vectors) {
        readDocIds(vectors - i);

        for (int j = 0; i < vectors; i++, j++) {
            int doc = docIdsScratch[j];

            if (acceptDocs == null || acceptDocs.get(doc)) {
                quantizeQueryIfNecessary();

                // Read single quantized vector and score
                float qcDist = osqVectorsScorer.quantizeScore(quantizedQueryScratch);

                // Read corrections for this vector
                indexInput.readFloats(correctiveValues, 0, 3);  // lower, upper, additional
                int quantizedComponentSum = Short.toUnsignedInt(indexInput.readShort());

                // Apply corrections
                float score = osqVectorsScorer.score(
                    queryCorrections.lowerInterval(),
                    queryCorrections.upperInterval(),
                    queryCorrections.quantizedComponentSum(),
                    queryCorrections.additionalCorrection(),
                    fieldInfo.getVectorSimilarityFunction(),
                    centroidDp,
                    correctiveValues[0],  // lowerInterval
                    correctiveValues[1],  // upperInterval
                    quantizedComponentSum,
                    correctiveValues[2],  // additionalCorrection
                    qcDist
                );

                scoredDocs++;
                knnCollector.collect(doc, score);
            } else {
                // Skip filtered doc
                indexInput.skipBytes(quantizedByteLength);
            }
        }
    }

    if (scoredDocs > 0) {
        knnCollector.incVisitedCount(scoredDocs);
    }

    return scoredDocs;
}
```

---

## Part 3: Reading Doc IDs

### Step A: `readDocIds()`

**File:** Same file
**Lines:** 711-718

```java
private void readDocIds(int count) throws IOException {
    // Read encoded doc IDs from IndexInput
    idsWriter.readInts(indexInput, count, docEncoding, docIdsScratch);

    // Convert deltas to absolute doc IDs
    for (int j = 0; j < count; j++) {
        docBase += docIdsScratch[j];      // docIdsScratch contains deltas
        docIdsScratch[j] = docBase;       // Store absolute doc ID
    }
}
```

**What happens:**
1. `idsWriter.readInts()` reads `count` doc IDs from the file
2. Doc IDs are stored as **deltas** (difference from previous ID)
3. Accumulate deltas to get absolute doc IDs
4. Example:
   ```
   Stored deltas:  [100, 5, 3, 10, ...]
   docBase:        0
   After reading:
     docBase = 0 + 100 = 100  → docIdsScratch[0] = 100
     docBase = 100 + 5 = 105  → docIdsScratch[1] = 105
     docBase = 105 + 3 = 108  → docIdsScratch[2] = 108
     docBase = 108 + 10 = 118 → docIdsScratch[3] = 118
   ```

---

## Part 4: Bulk SIMD Scoring

### Step D.1: `scoreBulk()` - Reading Vectors from Disk

**File:** `libs/simdvec/src/main/java/org/elasticsearch/simdvec/ESNextOSQVectorsScorer.java`
**Lines:** 220-257

```java
public float scoreBulk(
    byte[] q,                      // Quantized query
    float queryLowerInterval,
    float queryUpperInterval,
    int queryComponentSum,
    float queryAdditionalCorrection,
    VectorSimilarityFunction similarityFunction,
    float centroidDp,
    float[] scores                 // Output array
) throws IOException {
    // PHASE 1: Read and score quantized vectors (SIMD)
    quantizeScoreBulk(q, BULK_SIZE, scores);

    // PHASE 2: Read corrections (bulk arrays)
    in.readFloats(lowerIntervals, 0, BULK_SIZE);      // 16 floats
    in.readFloats(upperIntervals, 0, BULK_SIZE);      // 16 floats
    for (int i = 0; i < BULK_SIZE; i++) {
        targetComponentSums[i] = Short.toUnsignedInt(in.readShort());  // 16 shorts
    }
    in.readFloats(additionalCorrections, 0, BULK_SIZE);  // 16 floats

    // PHASE 3: Apply corrections to get final scores
    float maxScore = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < BULK_SIZE; i++) {
        scores[i] = score(
            queryLowerInterval,
            queryUpperInterval,
            queryComponentSum,
            queryAdditionalCorrection,
            similarityFunction,
            centroidDp,
            lowerIntervals[i],
            upperIntervals[i],
            targetComponentSums[i],
            additionalCorrections[i],
            scores[i]  // Quantized score from Phase 1
        );

        if (scores[i] > maxScore) {
            maxScore = scores[i];
        }
    }

    return maxScore;
}
```

### Step D.1.1: `quantizeScoreBulk()` - Reading Quantized Vectors

**Lines:** 140-168

```java
public void quantizeScoreBulk(byte[] q, int count, float[] scores) throws IOException {
    if (indexBits == 1) {
        if (queryBits == 4) {
            // Read and score 16 quantized vectors
            for (int i = 0; i < count; i++) {
                scores[i] = quantizeScore(q);  // Reads one vector from 'in'
            }
            return;
        }
    }
    // ... (other bit configurations)
}
```

### Step D.1.2: `quantizeScore()` - Reading ONE Quantized Vector

**Lines:** 103-133

This is where **the actual vector bytes are read from disk**:

```java
private long quantized4BitScore(byte[] q, int length) throws IOException {
    long subRet0 = 0;
    long subRet1 = 0;
    long subRet2 = 0;
    long subRet3 = 0;
    int r = 0;

    // Read in 8-byte (64-bit) chunks for SIMD efficiency
    for (final int upperBound = size & -Long.BYTES; r < upperBound; r += Long.BYTES) {
        final long value = in.readLong();  // ← READ VECTOR BYTES FROM DISK

        // Bit-level comparison with query (4 stripes for 4-bit query)
        subRet0 += Long.bitCount((long) BitUtil.VH_LE_LONG.get(q, r) & value);
        subRet1 += Long.bitCount((long) BitUtil.VH_LE_LONG.get(q, r + size) & value);
        subRet2 += Long.bitCount((long) BitUtil.VH_LE_LONG.get(q, r + 2 * size) & value);
        subRet3 += Long.bitCount((long) BitUtil.VH_LE_LONG.get(q, r + 3 * size) & value);
    }

    // Handle remaining bytes (4-byte chunks)
    for (final int upperBound = size & -Integer.BYTES; r < upperBound; r += Integer.BYTES) {
        final int value = in.readInt();  // ← READ VECTOR BYTES FROM DISK

        subRet0 += Integer.bitCount((int) BitUtil.VH_LE_INT.get(q, r) & value);
        subRet1 += Integer.bitCount((int) BitUtil.VH_LE_INT.get(q, r + size) & value);
        subRet2 += Integer.bitCount((int) BitUtil.VH_LE_INT.get(q, r + 2 * size) & value);
        subRet3 += Integer.bitCount((int) BitUtil.VH_LE_INT.get(q, r + 3 * size) & value);
    }

    // Handle tail bytes (single bytes)
    for (; r < size; r++) {
        final byte value = in.readByte();  // ← READ VECTOR BYTES FROM DISK

        subRet0 += Integer.bitCount((q[r] & value) & 0xFF);
        subRet1 += Integer.bitCount((q[r + size] & value) & 0xFF);
        subRet2 += Integer.bitCount((q[r + 2 * size] & value) & 0xFF);
        subRet3 += Integer.bitCount((q[r + 3 * size] & value) & 0xFF);
    }

    // Combine results from 4 stripes
    return subRet0 + (subRet1 << 1) + (subRet2 << 2) + (subRet3 << 3);
}
```

**What's happening:**
1. **`in.readLong()`** / **`in.readInt()`** / **`in.readByte()`** - These are reading the quantized vector bytes directly from the IndexInput (disk)
2. For each chunk, perform **bit-level XOR and popcount** (SIMD operations)
3. The 4-bit query is split into 4 "stripes" for parallel comparison
4. Result is a quantized distance score (needs corrections)

---

## Part 5: Individual Vector Scoring (< 50% pass filter)

### Step D.2: `scoreIndividually()`

**File:** `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsReader.java`
**Lines:** 646-688

```java
private float scoreIndividually() throws IOException {
    float maxScore = Float.NEGATIVE_INFINITY;

    // PHASE 1: Read and score quantized vectors one by one
    for (int j = 0; j < BULK_SIZE; j++) {
        int doc = docIdsScratch[j];

        if (doc != -1) {  // -1 means filtered out
            // Read quantized vector and compute quantized score
            float qcDist = osqVectorsScorer.quantizeScore(quantizedQueryScratch);
            scores[j] = qcDist;
        } else {
            // Skip this vector
            indexInput.skipBytes(quantizedVectorByteSize);
        }
    }

    // PHASE 2: Read ALL corrections in bulk (better cache locality)
    indexInput.readFloats(correctionsLower, 0, BULK_SIZE);
    indexInput.readFloats(correctionsUpper, 0, BULK_SIZE);
    for (int j = 0; j < BULK_SIZE; j++) {
        correctionsSum[j] = Short.toUnsignedInt(indexInput.readShort());
    }
    indexInput.readFloats(correctionsAdd, 0, BULK_SIZE);

    // PHASE 3: Apply corrections to get final scores
    for (int j = 0; j < BULK_SIZE; j++) {
        int doc = docIdsScratch[j];

        if (doc != -1) {
            scores[j] = osqVectorsScorer.score(
                queryCorrections.lowerInterval(),
                queryCorrections.upperInterval(),
                queryCorrections.quantizedComponentSum(),
                queryCorrections.additionalCorrection(),
                fieldInfo.getVectorSimilarityFunction(),
                centroidDp,
                correctionsLower[j],
                correctionsUpper[j],
                correctionsSum[j],
                correctionsAdd[j],
                scores[j]  // Quantized score
            );

            if (scores[j] > maxScore) {
                maxScore = scores[j];
            }
        }
    }

    return maxScore;
}
```

**Why individual scoring when < 50% pass filter?**
- If many docs are filtered out, reading all 16 vectors wastes I/O
- Individual scoring **skips filtered vectors** using `indexInput.skipBytes()`
- But still reads corrections in bulk for better cache performance

---

## Part 6: Complete Read Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ POSTING LIST FILE (.clivf)                                          │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ HEADER                                                          │ │
│  │  - rawCentroid[768]: float32 × 768 = 3072 bytes               │ │
│  │  - centroidDp: float32 = 4 bytes                               │ │
│  │  - vectorCount: VInt = 1-5 bytes                               │ │
│  │  - docIdEncoding: byte = 1 byte                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│         ↓ resetPostingsScorer() reads header                        │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ BULK BLOCK 0 (16 vectors)                                      │ │
│  │  ┌──────────────────────────────────────────────────────────┐ │ │
│  │  │ Doc IDs (delta-encoded)                                   │ │
│  │  │  - Stored inline or separately depending on encoding      │ │
│  │  └──────────────────────────────────────────────────────────┘ │ │
│  │         ↓ readDocIds(16) reads and reconstructs absolute IDs    │ │
│  │                                                                  │ │
│  │  ┌──────────────────────────────────────────────────────────┐ │ │
│  │  │ QUANTIZED VECTORS (sequential, 16 vectors)                │ │
│  │  │  vector[0]:  96 bytes (768 dims × 1 bit / 8)             │ │
│  │  │  vector[1]:  96 bytes                                     │ │
│  │  │  ...                                                       │ │
│  │  │  vector[15]: 96 bytes                                     │ │
│  │  │  TOTAL: 1536 bytes                                        │ │
│  │  └──────────────────────────────────────────────────────────┘ │ │
│  │         ↓ quantizeScoreBulk() OR scoreIndividually()            │ │
│  │         ↓ Reads via in.readLong() / in.readInt() / in.readByte()│ │
│  │                                                                  │ │
│  │  ┌──────────────────────────────────────────────────────────┐ │ │
│  │  │ CORRECTIONS (bulk arrays)                                 │ │
│  │  │  lowerIntervals[16]:        64 bytes (16 × 4)            │ │
│  │  │  upperIntervals[16]:        64 bytes (16 × 4)            │ │
│  │  │  componentSums[16]:         32 bytes (16 × 2)            │ │
│  │  │  additionalCorrections[16]: 64 bytes (16 × 4)            │ │
│  │  │  TOTAL: 224 bytes                                         │ │
│  │  └──────────────────────────────────────────────────────────┘ │ │
│  │         ↓ in.readFloats() and in.readShort() read corrections   │ │
│  │         ↓ score() applies corrections to quantized scores       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│         ↓ collectBulk() sends scores to KnnCollector                │
│                                                                      │
│  [BULK BLOCK 1...N-1] - Repeat above                                │
│                                                                      │
│  [TAIL BLOCK] - Individual scoring for remaining < 16 vectors       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 7: Read Pattern Summary

### Sequential Reading

**Order:**
1. **Posting List Header** → `resetPostingsScorer()` (lines 629-688)
2. **For each bulk block (16 vectors):**
   - **Doc IDs** → `readDocIds()` (lines 711-718)
   - **Quantized Vectors** → `quantizeScore()` via `in.readLong()/readInt()/readByte()` (ESNextOSQVectorsScorer.java:103-133)
   - **Corrections** → `in.readFloats()` and `in.readShort()` (ESNextOSQVectorsScorer.java:231-236)
3. **Tail vectors** → Individual reads for remaining < 16 vectors

### Why Sequential?

1. **Disk locality** - Reading contiguous bytes is fastest
2. **Cache efficiency** - Sequential reads maximize CPU cache hits
3. **Prefetching** - OS and disk controllers can predict next reads
4. **Bulk operations** - Process 16 vectors at once for SIMD

---

## Part 8: Two Scoring Strategies

### Strategy A: Bulk SIMD (≥ 50% pass filter)

**When:** Most documents pass the filter (`docsToBulkScore >= BULK_SIZE / 2`)

**Process:**
```
1. Read ALL 16 quantized vectors sequentially
2. Read ALL 16 sets of corrections
3. Score all 16 using SIMD operations
4. Collect only those passing filter
```

**Advantage:** Maximum SIMD efficiency (16 vectors at once)

**Cost:** Reads filtered vectors too (but acceptable if < 50% filtered)

### Strategy B: Individual (< 50% pass filter)

**When:** Many documents filtered out (`docsToBulkScore < BULK_SIZE / 2`)

**Process:**
```
1. For each of 16 vectors:
   - If filtered: SKIP vector (indexInput.skipBytes())
   - If passes: READ and score vector
2. Read ALL corrections in bulk (better cache)
3. Apply corrections
```

**Advantage:** Avoids reading filtered vectors (saves I/O when heavily filtered)

**Cost:** Can't use full SIMD (but many docs filtered anyway)

---

## Part 9: Example Read Trace

Let's trace reading 16 vectors from disk (768-dim, 1-bit quantization):

```
File pointer: 16384 (start of bulk block 0)

1. readDocIds(16)
   Read: delta-encoded doc IDs (16-64 bytes depending on encoding)
   Output: docIdsScratch = [100, 105, 108, 110, 115, ...]
   File pointer: 16448

2. Pre-filter check
   acceptDocs.get(100) = true
   acceptDocs.get(105) = false  ← filtered
   acceptDocs.get(108) = true
   ...
   Result: 13 of 16 pass filter (> 50%)
   → Use BULK SIMD strategy

3. quantizeQueryIfNecessary()
   Quantize query vector once (not reading from disk)

4. quantizeScoreBulk(q, 16, scores)
   For each vector 0-15:
     in.readLong()  ← Read 8 bytes of quantized vector
     in.readLong()  ← Read next 8 bytes
     ...
     in.readLong()  ← Read last 8 bytes
     (Total: 12 × readLong() = 96 bytes per vector)
     (16 vectors = 1536 bytes)
   File pointer: 17984

5. Read corrections (bulk)
   in.readFloats(lowerIntervals, 0, 16)  → Read 64 bytes
   in.readFloats(upperIntervals, 0, 16)  → Read 64 bytes
   For i in 0..15:
     in.readShort()                      → Read 2 bytes each (32 total)
   in.readFloats(additionalCorr, 0, 16)  → Read 64 bytes
   File pointer: 18208 (224 bytes)

6. Apply corrections
   For each of 16 vectors:
     scores[i] = score(query params, corrections[i], scores[i])

7. Collect results
   For i in 0..15:
     If docIdsScratch[i] != -1:  ← Not filtered
       knnCollector.collect(docIdsScratch[i], scores[i])

Total read for 16 vectors: ~1760 bytes (doc IDs + vectors + corrections)
```

---

## Part 10: Key Takeaways

1. **Vectors stored on disk:** `.clivf` file, quantized, in 16-vector blocks
2. **Reading is sequential:** One posting list at a time, one block at a time
3. **IndexInput is the reader:** Low-level Lucene interface for file I/O
4. **Two strategies:** Bulk SIMD (fast, reads all) vs Individual (I/O efficient, skips filtered)
5. **Read order:** Header → Doc IDs → Quantized Vectors → Corrections → Score → Collect
6. **SIMD happens in memory:** After reading quantized bytes, bit operations are CPU-only
7. **Corrections read separately:** After quantized vectors for better cache locality

## Code References

- **visit()**: `server/.../ESNextDiskBBQVectorsReader.java:721-793`
- **readDocIds()**: Same file, lines 711-718
- **scoreIndividually()**: Same file, lines 646-688
- **scoreBulk()**: `libs/simdvec/.../ESNextOSQVectorsScorer.java:220-257`
- **quantizeScore()**: Same file, lines 103-133 (actual disk reads)
- **resetPostingsScorer()**: ESNextDiskBBQVectorsReader.java:629-688

---

## Summary

**Where are vectors read?**
- From the `.clivf` file via `IndexInput`
- In the `quantizeScore()` method using `in.readLong()` / `in.readInt()` / `in.readByte()`

**How are they processed?**
1. Seek to posting list offset
2. Read header (centroid, count, encoding)
3. For each 16-vector block:
   - Read doc IDs (delta-encoded)
   - Read quantized vectors (1-7 bits, packed bytes)
   - Read corrections (float32 arrays)
   - Compute scores (XOR + popcount + corrections)
   - Collect results above threshold
4. Process tail individually
5. Move to next posting list

This sequential, block-oriented approach maximizes disk throughput and CPU SIMD utilization!
