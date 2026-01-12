# Elasticsearch Vector Search: Quantization System

## Overview

This document provides a comprehensive analysis of quantization in Elasticsearch's vector search system, covering what gets quantized, when quantization occurs, how it works algorithmically, and whether stored values are quantized.

## Table of Contents
1. [What Gets Quantized](#what-gets-quantized)
2. [When Quantization Happens](#when-quantization-happens)
3. [How Quantization Works](#how-quantization-works)
4. [Stored Values: Quantized or Raw](#stored-values-quantized-or-raw)
5. [Quantization Types](#quantization-types)
6. [Scoring with Quantized Vectors](#scoring-with-quantized-vectors)
7. [Key Code References](#key-code-references)

---

## What Gets Quantized

### Elements Subject to Quantization

1. **Indexed Document Vectors**
   - Original float32 vectors (768-2048 dimensions)
   - Quantized to 1-7 bits per dimension
   - **File:** `server/src/main/java/org/elasticsearch/index/codec/vectors/es818/ES818BinaryQuantizedVectorsWriter.java`

2. **Query Vectors**
   - User-provided search queries
   - Quantized at search time using same algorithm
   - Typically uses higher precision than index (e.g., 4-bit for binary quantization)
   - **File:** `server/src/main/java/org/elasticsearch/index/codec/vectors/es816/BinaryQuantizer.java`

3. **Centroid Vectors (IVF)**
   - Cluster center vectors
   - Quantized using 7-bit scalar quantization
   - Stored separately from document vectors
   - **File:** `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsWriter.java`

4. **NOT Quantized:**
   - Raw centroid vectors in posting lists (stored as float32)
   - Global centroid (mean of all vectors)
   - Correction metadata (stored as float32)

---

## When Quantization Happens

### Indexing Time (Write Path)

#### 1. Scalar Quantization (OSQ)

**Location:** `server/src/main/java/org/elasticsearch/index/codec/vectors/es818/ES818BinaryQuantizedVectorsWriter.java:145-175`

```java
@Override
public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    // Step 1: Write raw vectors first (delegate to Lucene99)
    rawVectorDelegate.flush(maxDoc, sortMap);

    // Step 2: Normalize if using COSINE similarity
    if (fieldInfo.getVectorSimilarityFunction() == COSINE) {
        normalizeVectors(fieldData.vectors);
    }

    // Step 3: Calculate centroid (mean of all vectors)
    float[] centroid = new float[dimension];
    for (float[] vector : fieldData.vectors) {
        ESVectorUtil.add(centroid, vector);
    }
    ESVectorUtil.divInPlace(centroid, fieldData.vectors.size());

    // Step 4: Create quantizer
    OptimizedScalarQuantizer scalarQuantizer = new OptimizedScalarQuantizer();

    // Step 5: Quantize each vector
    writeBinarizedVectors(fieldData, centroid, scalarQuantizer);
}
```

**Detailed Quantization per Vector:**

```java
// ES818BinaryQuantizedVectorsWriter.java:196-219
private void writeBinarizedVectors(
    FieldWriter fieldData,
    float[] clusterCenter,
    OptimizedScalarQuantizer scalarQuantizer
) throws IOException {
    int discreteDims = BQVectorUtils.discretize(dimension, 64);
    int[] quantizationScratch = new int[discreteDims];
    byte[] vector = new byte[discreteDims / 8];
    float[] scratch = new float[dimension];

    for (int i = 0; i < fieldData.getVectors().size(); i++) {
        float[] v = fieldData.getVectors().get(i);

        // Quantize individual vector to 1-bit
        OptimizedScalarQuantizer.QuantizationResult corrections = scalarQuantizer.scalarQuantize(
            v,                      // Input vector
            scratch,                // Residual destination
            quantizationScratch,    // Quantized destination
            (byte) 1,               // 1-bit quantization
            clusterCenter           // Reference point
        );

        // Pack as binary (1 bit per dimension)
        ESVectorUtil.packAsBinary(quantizationScratch, vector);

        // Write quantized vector
        binarizedVectorData.writeBytes(vector, vector.length);

        // Write correction metadata
        binarizedVectorData.writeInt(Float.floatToIntBits(corrections.lowerInterval()));
        binarizedVectorData.writeInt(Float.floatToIntBits(corrections.upperInterval()));
        binarizedVectorData.writeInt(Float.floatToIntBits(corrections.additionalCorrection()));
        binarizedVectorData.writeShort((short) corrections.quantizedComponentSum());
    }
}
```

#### 2. IVF Posting List Quantization

**Location:** `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/DiskBBQBulkWriter.java:71-180`

```java
public void writeBulk(float[][] vectors, int[] docIds, int offset, int count) throws IOException {
    // For each vector in bulk (up to 32)
    for (int i = 0; i < count; i++) {
        float[] vector = vectors[offset + i];

        // Center relative to centroid
        ESVectorUtil.subtract(vector, centroid, scratchVector);

        // Quantize (1, 2, 4, or 7 bits)
        OptimizedScalarQuantizer.QuantizationResult result = quantizer.scalarQuantize(
            scratchVector,      // Centered vector
            residual,           // Residual output
            quantizedScratch,   // Quantized output
            bits,               // Bit precision (1/2/4/7)
            null                // Already centered
        );

        // Write quantized vector
        idsWriter.writeInts(output, dimension, quantEncoding, quantizedScratch);

        // Buffer corrections for bulk write
        correctionsLower[i] = result.lowerInterval();
        correctionsUpper[i] = result.upperInterval();
        correctionsSum[i] = (short) result.quantizedComponentSum();
        correctionsAdd[i] = result.additionalCorrection();
    }

    // Write all corrections in bulk (better cache locality)
    for (int i = 0; i < count; i++) {
        output.writeInt(Float.floatToIntBits(correctionsLower[i]));
    }
    // ... (write other corrections)
}
```

### Search Time (Read Path)

#### Query Quantization

**Location:** `server/src/main/java/org/elasticsearch/index/codec/vectors/es816/ES816BinaryFlatVectorsScorer.java:54-79`

```java
@Override
public RandomVectorScorer getRandomVectorScorer(
    VectorSimilarityFunction similarityFunction,
    KnnVectorValues vectorValues,
    float[] target  // Query vector
) throws IOException {
    if (vectorValues instanceof BinarizedByteVectorValues binarizedVectors) {
        BinaryQuantizer quantizer = binarizedVectors.getQuantizer();
        float[] centroid = binarizedVectors.getCentroid();

        // Quantize query vector (4-bit for binary quantization)
        int discretizedDimensions = BQVectorUtils.discretize(target.length, 64);
        byte[] quantized = new byte[BinaryQuantizer.B_QUERY * discretizedDimensions / 8];

        BinaryQuantizer.QueryFactors factors = quantizer.quantizeForQuery(
            target,      // Query vector
            quantized,   // Destination buffer
            centroid     // Reference centroid
        );

        BinaryQueryVector queryVector = new BinaryQueryVector(quantized, factors);
        return new BinarizedRandomVectorScorer(queryVector, binarizedVectors, similarityFunction);
    }

    // Fallback to non-quantized scoring
    return nonQuantizedDelegate.getRandomVectorScorer(similarityFunction, vectorValues, target);
}
```

---

## How Quantization Works

### Optimized Scalar Quantization (OSQ)

**Primary Implementation:** `server/src/main/java/org/elasticsearch/index/codec/vectors/OptimizedScalarQuantizer.java:19-219`

#### Algorithm Overview

OSQ uses coordinate descent optimization to minimize quantization loss while mapping float values to integer bins.

#### Key Data Structure

```java
// OptimizedScalarQuantizer.java:60-109
public record QuantizationResult(
    float lowerInterval,        // Minimum quantization boundary
    float upperInterval,        // Maximum quantization boundary
    float additionalCorrection, // Similarity-specific correction
    int quantizedComponentSum   // Sum of all quantized values
) {}
```

#### Main Quantization Method

```java
// OptimizedScalarQuantizer.java:110-164
public QuantizationResult scalarQuantize(
    float[] vector,              // Input vector
    float[] residualDestination, // Centered vector output
    int[] destination,           // Quantized output
    byte bits,                   // Precision (1-8 bits)
    float[] centroid             // Reference point (can be null)
) {
    int dimension = vector.length;

    // Step 1: Center vector and calculate statistics
    ESVectorUtil.centerAndCalculateOSQStatsEuclidean(
        vector,
        centroid,
        residualDestination,  // Output: vector - centroid
        statsScratch          // Output: [mean, variance, min, max, norm2, dotProd]
    );

    float vecMean = statsScratch[0];
    float vecVar = statsScratch[1];
    float min = statsScratch[2];
    float max = statsScratch[3];
    float norm2 = statsScratch[4];
    float vecStd = (float) Math.sqrt(vecVar);

    // Step 2: Initialize quantization interval using MSE grid
    initInterval(bits, vecStd, vecMean, min, max, intervalScratch);
    // intervalScratch[0] = lowerInterval
    // intervalScratch[1] = upperInterval

    // Step 3: Optimize intervals via coordinate descent
    int points = 1 << bits;  // 2^bits quantization levels
    boolean hasQuantization = optimizeIntervals(
        intervalScratch,        // In/out: interval bounds
        destination,            // Out: quantized values
        residualDestination,    // In: centered vector
        norm2,                  // Vector norm
        points                  // Number of quantization levels
    );

    if (!hasQuantization) {
        // Degenerate case: all values same
        Arrays.fill(destination, 0);
        return new QuantizationResult(0, 0, norm2, 0);
    }

    // Step 4: Final quantization with optimized intervals
    int sumQuery = ESVectorUtil.quantizeVectorWithIntervals(
        residualDestination,
        destination,
        intervalScratch[0],  // lowerInterval
        intervalScratch[1],  // upperInterval
        bits
    );

    return new QuantizationResult(
        intervalScratch[0],
        intervalScratch[1],
        similarityFunction == EUCLIDEAN ? norm2 : statsScratch[5],
        sumQuery
    );
}
```

#### Step 1: Centering and Statistics

```java
// ESVectorUtil.java:220-250
public static void centerAndCalculateOSQStatsEuclidean(
    float[] vector,
    float[] centroid,
    float[] residual,
    float[] stats  // [mean, variance, min, max, norm2, dotProd]
) {
    double sum = 0.0;
    double sumSq = 0.0;
    float min = Float.POSITIVE_INFINITY;
    float max = Float.NEGATIVE_INFINITY;
    float dotProd = 0.0f;

    for (int i = 0; i < vector.length; i++) {
        // Center: residual = vector - centroid
        float val = centroid == null ? vector[i] : vector[i] - centroid[i];
        residual[i] = val;

        // Accumulate statistics
        sum += val;
        sumSq += val * val;
        min = Math.min(min, val);
        max = Math.max(max, val);

        if (centroid != null) {
            dotProd += vector[i] * centroid[i];
        }
    }

    int n = vector.length;
    float mean = (float) (sum / n);
    float variance = (float) (sumSq / n - mean * mean);

    stats[0] = mean;
    stats[1] = variance;
    stats[2] = min;
    stats[3] = max;
    stats[4] = (float) sumSq;  // norm2
    stats[5] = dotProd;
}
```

#### Step 2: Interval Initialization

Uses pre-computed MSE-optimal grids for different bit sizes:

```java
// OptimizedScalarQuantizer.java:20-23
private static final float[] MSE_1BIT_GRID = {-0.43f, 0.43f};
private static final float[] MSE_2BIT_GRID = {-1.15f, -0.31f, 0.31f, 1.15f};
private static final float[] MSE_4BIT_GRID = {...};  // 16 values
private static final float[] MSE_7BIT_GRID = {...};  // 128 values
```

```java
// OptimizedScalarQuantizer.java:165-185
private void initInterval(
    byte bits,
    float vecStd,
    float vecMean,
    float min,
    float max,
    float[] intervalDest
) {
    float[] grid = switch (bits) {
        case 1 -> MSE_1BIT_GRID;
        case 2 -> MSE_2BIT_GRID;
        case 4 -> MSE_4BIT_GRID;
        case 7 -> MSE_7BIT_GRID;
        default -> throw new IllegalArgumentException("Unsupported bits: " + bits);
    };

    // Scale grid by standard deviation and shift by mean
    intervalDest[0] = grid[0] * vecStd + vecMean;           // lower
    intervalDest[1] = grid[grid.length - 1] * vecStd + vecMean;  // upper

    // Ensure interval contains [min, max]
    intervalDest[0] = Math.min(intervalDest[0], min);
    intervalDest[1] = Math.max(intervalDest[1], max);
}
```

#### Step 3: Coordinate Descent Optimization

```java
// OptimizedScalarQuantizer.java:187-219
private boolean optimizeIntervals(
    float[] interval,        // [lower, upper]
    int[] quantized,         // Output quantized values
    float[] centered,        // Centered vector
    float norm2,             // Vector norm squared
    int numPoints            // 2^bits
) {
    final int MAX_ITERATIONS = 5;
    final float LAMBDA = 0.1f;  // Smoothing parameter

    float prevLoss = Float.POSITIVE_INFINITY;

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        // Quantize with current intervals
        ESVectorUtil.quantizeVectorWithIntervals(
            centered,
            quantized,
            interval[0],
            interval[1],
            (byte) (31 - Integer.numberOfLeadingZeros(numPoints - 1))  // log2(numPoints)
        );

        // Calculate reconstruction loss
        float loss = calculateLoss(centered, quantized, interval, norm2, numPoints);

        // Early exit if loss increases
        if (loss > prevLoss) {
            break;
        }

        // Update intervals to minimize loss
        updateIntervals(centered, quantized, interval, numPoints, LAMBDA);

        prevLoss = loss;
    }

    // Check if quantization produced variation
    int first = quantized[0];
    for (int q : quantized) {
        if (q != first) {
            return true;  // Has variation
        }
    }

    return false;  // Degenerate (all same value)
}
```

#### Step 4: Final Quantization

```java
// ESVectorUtil.java:410-445
public static int quantizeVectorWithIntervals(
    float[] vector,
    int[] destination,
    float lowerInterval,
    float upperInterval,
    byte bits
) {
    int numPoints = 1 << bits;  // 2^bits
    float range = upperInterval - lowerInterval;
    float scale = (numPoints - 1) / range;

    int sum = 0;

    for (int i = 0; i < vector.length; i++) {
        float val = vector[i];

        // Map to quantization level
        float normalized = (val - lowerInterval) * scale;
        int quantized = Math.round(normalized);

        // Clamp to [0, numPoints-1]
        quantized = Math.max(0, Math.min(numPoints - 1, quantized));

        destination[i] = quantized;
        sum += quantized;
    }

    return sum;
}
```

**Example (1-bit):**
```
Input:  [-0.8, -0.2, 0.1, 0.6, 0.9]
Centered: [-0.3, 0.3, 0.6, 1.1, 1.4]  // Subtract mean 0.5

Interval: [-0.43, 0.43] (from MSE grid)
Quantization:
  -0.3: ((-0.3 - (-0.43)) / 0.86) * 1 = 0.15 → 0
   0.3: (( 0.3 - (-0.43)) / 0.86) * 1 = 0.85 → 1
   0.6: (( 0.6 - (-0.43)) / 0.86) * 1 = 1.20 → 1
   1.1: (( 1.1 - (-0.43)) / 0.86) * 1 = 1.78 → 1
   1.4: (( 1.4 - (-0.43)) / 0.86) * 1 = 2.13 → 1 (clamped)

Output: [0, 1, 1, 1, 1]
```

---

### Binary Quantization (RaBitQ)

**Implementation:** `server/src/main/java/org/elasticsearch/index/codec/vectors/es816/BinaryQuantizer.java:37-385`

#### Algorithm Overview

RaBitQ (Radial Basis Quantization) reduces each dimension to a single bit (sign) relative to a centroid. This achieves maximum compression (~32× smaller than float32).

**Paper:** "RaBitQ: Quantizing High-Dimensional Vectors with a Radial Basis Function"
**URL:** https://arxiv.org/abs/2405.12497

#### Index Quantization (1-bit)

```java
// BinaryQuantizer.java:247-282
public IndexCorrections quantizeForIndex(
    float[] vector,
    byte[] destination,  // 1 bit per dimension packed
    float[] centroid
) {
    // Step 1: Calculate distances
    float distToC = VectorUtil.squareDistance(vector, centroid);
    float vDotC = VectorUtil.dotProduct(vector, centroid);

    // Step 2: Center vector
    BQVectorUtils.subtractInPlace(vector, centroid);
    float normVmC = BQVectorUtils.norm(vector);

    // Step 3: Quantize to binary (sign bit only)
    int[] discretized = BQVectorUtils.discretize(vector, discretizedDimensions);
    ESVectorUtil.packAsBinary(discretized, destination);

    // Step 4: Calculate correction factors
    float cDotC = VectorUtil.dotProduct(centroid, centroid);
    float ooq = cDotC + normVmC * normVmC - 2 * vDotC;  // ||v - c||²
    float normOC = VectorUtil.squareDistance(vector, centroid);
    float oDotC = VectorUtil.dotProduct(vector, centroid);

    return new IndexCorrections(ooq, normOC, oDotC, centroid);
}
```

#### Query Quantization (4-bit)

```java
// BinaryQuantizer.java:323-384
public QueryFactors quantizeForQuery(
    float[] query,
    byte[] destination,  // 4 bits per dimension packed
    float[] centroid
) {
    // Step 1: Center query
    float qDotC = VectorUtil.dotProduct(query, centroid);
    BQVectorUtils.subtractInPlace(query, centroid);

    // Step 2: Calculate range for 4-bit quantization
    float[] range = range(query);  // [min, max]
    float lower = range[0];
    float width = (range[1] - lower) / ((1 << B_QUERY) - 1);  // (2^4 - 1) = 15

    // Step 3: Quantize to 4-bit values
    QuantResult quantResult = quantize(query, lower, width);
    int[] quantized = quantResult.result();

    // Step 4: Transpose for efficient comparison (half-byte packing)
    ESVectorUtil.transposeHalfByte(quantized, destination);

    // Step 5: Calculate query factors for scoring
    float qcDist = VectorUtil.squareDistance(query, centroid);
    float cDotC = VectorUtil.dotProduct(centroid, centroid);

    return new QueryFactors(
        lower,           // Lower quantization bound
        width,           // Quantization step size
        quantResult.bias(),  // Quantization bias
        qcDist,          // ||q - c||²
        qDotC,           // q·c
        cDotC            // c·c
    );
}
```

**4-bit Quantization Detail:**

```java
// BinaryQuantizer.java:386-420
private QuantResult quantize(float[] vector, float lower, float width) {
    int[] quantized = new int[vector.length];
    int bias = 0;

    for (int i = 0; i < vector.length; i++) {
        // Map to [0, 15] range
        float normalized = (vector[i] - lower) / width;
        int q = Math.round(normalized);

        // Clamp to 4-bit range [0, 15]
        q = Math.max(0, Math.min(15, q));

        quantized[i] = q;
        bias += q;
    }

    return new QuantResult(quantized, bias);
}
```

**Bit Packing:**

```java
// ESVectorUtil.java:490-520
public static void packAsBinary(int[] discretized, byte[] destination) {
    // Pack 8 bits per byte
    for (int i = 0; i < destination.length; i++) {
        int byteVal = 0;
        for (int bit = 0; bit < 8; bit++) {
            int idx = i * 8 + bit;
            if (idx < discretized.length && discretized[idx] > 0) {
                byteVal |= (1 << (7 - bit));
            }
        }
        destination[i] = (byte) byteVal;
    }
}

public static void transposeHalfByte(int[] quantized, byte[] destination) {
    // Pack 2 values per byte (4 bits each)
    for (int i = 0; i < quantized.length; i += 2) {
        int high = quantized[i] & 0xF;
        int low = (i + 1 < quantized.length) ? (quantized[i + 1] & 0xF) : 0;
        destination[i / 2] = (byte) ((high << 4) | low);
    }
}
```

---

## Stored Values: Quantized or Raw?

### What Is Stored

#### Quantized Components (Stored)

1. **Indexed Vectors (Quantized)**
   - **Storage:** `.clivf` file (posting lists)
   - **Format:** 1-7 bits per dimension, packed into bytes
   - **Size:** `(dimension * bits) / 8` bytes per vector
   - **Example (768-dim, 1-bit):** 96 bytes per vector (vs 3072 bytes for float32)

2. **Centroid Vectors (Quantized)**
   - **Storage:** `.cenivf` file
   - **Format:** 7-bit quantization
   - **Purpose:** Fast centroid scoring during search

3. **Correction Metadata (Raw float32)**
   - **Storage:** Alongside quantized vectors
   - **Components:**
     - `lowerInterval` (4 bytes)
     - `upperInterval` (4 bytes)
     - `quantizedComponentSum` (2 bytes)
     - `additionalCorrection` (4 bytes)
   - **Total:** 14 bytes per vector

#### Raw Components (Stored)

1. **Raw Centroid in Posting Lists**
   - **Storage:** `.clivf` file, at start of each posting list
   - **Format:** Full precision float32
   - **Size:** `dimension * 4` bytes
   - **Purpose:** Accurate distance computation for scoring

2. **Global Centroid**
   - **Storage:** `.mivf` metadata file
   - **Format:** Full precision float32
   - **Purpose:** Reference point for centering queries

#### NOT Stored

- **Original float32 vectors are NOT stored** in quantized formats
- For formats requiring raw vectors (e.g., HNSW graph traversal), a delegate format is used:

```java
// ES814ScalarQuantizedVectorsFormat.java:51-57
@Override
public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    // Return raw vectors from delegate (not quantized storage)
    return rawVectorDelegate.getFloatVectorValues(field);
}
```

### Storage Size Comparison

**Example: 768-dimensional vector**

| Format | Size per Vector | Compression Ratio |
|--------|----------------|-------------------|
| Float32 (raw) | 3,072 bytes | 1.0× (baseline) |
| 7-bit quantized | 672 bytes + 14 bytes = 686 bytes | 4.5× |
| 4-bit quantized | 384 bytes + 14 bytes = 398 bytes | 7.7× |
| 2-bit quantized | 192 bytes + 14 bytes = 206 bytes | 14.9× |
| 1-bit quantized | 96 bytes + 14 bytes = 110 bytes | 27.9× |

**Actual disk usage includes:**
- Quantized vector bytes
- Correction metadata (14 bytes)
- Doc ID (1-4 bytes, delta-encoded)
- Raw centroid per posting list (3,072 bytes, amortized)

---

## Quantization Types

### Type 1: Scalar Quantization (OSQ)

**Purpose:** General-purpose quantization with tunable precision

**Implementations:**
- `ES814ScalarQuantizedVectorsFormat` (4-bit and 7-bit)
- `ES93ScalarQuantizedVectorsFormat` (4-bit and 7-bit)
- `ESNextDiskBBQVectorsFormat` (1, 2, 4, 7-bit for IVF)

**Characteristics:**
- Each dimension quantized independently
- Coordinate descent optimization
- Works with all similarity functions (DOT_PRODUCT, COSINE, EUCLIDEAN)
- Correction factors enable accurate similarity reconstruction

**Use Cases:**
- General vector search where balance of speed/accuracy needed
- High-dimensional vectors (768-2048 dims)
- When 4-7 bits provides sufficient recall

**Code References:**
- **Core:** `server/src/main/java/org/elasticsearch/index/codec/vectors/OptimizedScalarQuantizer.java`
- **ES814:** `server/src/main/java/org/elasticsearch/index/codec/vectors/ES814ScalarQuantizedVectorsFormat.java:46-307`
- **ES93:** `server/src/main/java/org/elasticsearch/index/codec/vectors/es93/ES93ScalarQuantizedVectorsFormat.java:42-100`

### Type 2: Binary Quantization (RaBitQ)

**Purpose:** Maximum compression for billion-scale vector search

**Implementations:**
- `ES816BinaryQuantizedVectorsFormat`
- `ES818BinaryQuantizedVectorsFormat`

**Characteristics:**
- 1-bit per dimension for index vectors
- 4-bit per dimension for query vectors
- Radial basis function approach
- Fast bit-level operations (XOR, popcount)
- Asymmetric quantization (different for index vs query)

**Use Cases:**
- Very large scale (billions of vectors)
- When disk/memory is primary constraint
- Normalized vectors (COSINE similarity)
- Acceptable recall with ~32× compression

**Code References:**
- **Core:** `server/src/main/java/org/elasticsearch/index/codec/vectors/es816/BinaryQuantizer.java:37-385`
- **ES818:** `server/src/main/java/org/elasticsearch/index/codec/vectors/es818/ES818BinaryQuantizedVectorsFormat.java`

### Type 3: IVF with Multi-bit Quantization

**Purpose:** Disk-efficient vector search with IVF clustering

**Implementation:**
- `ESNextDiskBBQVectorsFormat`

**Characteristics:**
- Hierarchical k-means clustering
- Variable bit quantization (1, 2, 4, 7-bit)
- Vectors quantized relative to cluster centroid
- Two-level search (centroid → posting lists)

**Use Cases:**
- Disk-backed vector search
- When index size exceeds memory
- Need for both speed and accuracy
- Billion-scale with controlled memory usage

**Code References:**
- **Format:** `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsFormat.java`
- **Writer:** `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsWriter.java`
- **Reader:** `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsReader.java`

### Quantization Encoding Options

```java
// ESNextDiskBBQVectorsFormat.java:76-209
public enum QuantEncoding {
    ONE_BIT_4BIT_QUERY(0, 1, 4),        // Extreme compression
    TWO_BIT_4BIT_QUERY(1, 2, 4),        // High compression
    FOUR_BIT_SYMMETRIC(2, 4, 4);        // Balanced

    private final int id;
    private final int docBits;    // Bits for index vectors
    private final int queryBits;  // Bits for query vectors

    public int getDocPackedLength(int dimension) {
        int discretized = discretizedDimensions(dimension);
        return (discretized * docBits + 7) / 8;  // Round up to bytes
    }
}
```

---

## Scoring with Quantized Vectors

### OSQ Score Reconstruction

**Location:** `libs/simdvec/src/main/java/org/elasticsearch/simdvec/ES91OSQVectorsScorer.java:97-131`

#### Algorithm

The scoring process reconstructs approximate similarity from:
1. **Quantized distance** (from bit operations)
2. **Interval corrections** (lower/upper bounds)
3. **Component sums** (accumulated quantized values)
4. **Additional corrections** (norm and centroid adjustments)

```java
// ES91OSQVectorsScorer.java:97-131
public float score(
    float queryLowerInterval,
    float queryUpperInterval,
    int queryComponentSum,
    float queryAdditionalCorrection,
    VectorSimilarityFunction similarityFunction,
    float centroidDp,
    float lowerInterval,
    float upperInterval,
    int targetComponentSum,
    float additionalCorrection,
    float qcDist  // Quantized distance from bit operations
) {
    // Reconstruct similarity from quantized approximation
    float ax = lowerInterval;
    float lx = upperInterval - ax;
    float ay = queryLowerInterval;
    float ly = (queryUpperInterval - ay) * FOUR_BIT_SCALE;  // 1/15 for 4-bit query
    float y1 = queryComponentSum;
    int dimensions = getDimensions();

    // Bilinear interpolation of quantized distance
    float score = ax * ay * dimensions              // Lower bounds product
                + ay * lx * targetComponentSum      // Lower × upper scaling
                + ax * ly * y1                      // Upper × lower scaling
                + lx * ly * qcDist;                 // Upper bounds × distance

    // Apply similarity-specific corrections
    if (similarityFunction == EUCLIDEAN) {
        // Euclidean: convert to distance, then similarity
        score = queryAdditionalCorrection + additionalCorrection - 2 * score;
        return Math.max(1 / (1f + score), 0);
    } else {
        // Dot product or cosine: add corrections
        score += queryAdditionalCorrection + additionalCorrection - centroidDp;

        if (similarityFunction == MAXIMUM_INNER_PRODUCT) {
            return VectorUtil.scaleMaxInnerProductScore(score);
        }

        // Cosine similarity: normalize to [0, 1]
        return Math.max((1f + score) / 2f, 0);
    }
}
```

#### Quantized Distance Computation (SIMD)

```java
// ES91OSQVectorsScorer.java:51-81
public int quantizeScore(byte[] query) {
    // XOR bits and count matches (SIMD optimized)
    int matches = 0;
    for (int i = 0; i < query.length; i += SIMD_WIDTH) {
        // Use SIMD XOR + popcount
        long q = readLong(query, i);
        long d = readLong(docVector, i);
        long xor = q ^ d;
        matches += Long.bitCount(xor);
    }
    return matches;
}
```

### Binary Quantization Scoring

**Location:** `server/src/main/java/org/elasticsearch/index/codec/vectors/es816/ES816BinaryFlatVectorsScorer.java:98-200`

```java
// ES816BinaryFlatVectorsScorer.java:143-189
public float score(BinaryQueryVector query, BinarizedByteVectorValues docVector) {
    // Step 1: Bit-level inner product
    int dotProduct = innerProduct(query.vector(), docVector.vector());

    // Step 2: Apply corrections
    IndexCorrections docCorr = docVector.getCorrections();
    QueryFactors queryFactors = query.getFactors();

    // Reconstruct similarity
    float score = dotProduct * queryFactors.width() * docVector.getScale();
    score += queryFactors.bias() * docVector.getBias();
    score += queryFactors.qDotC();
    score += docCorr.oDotC();
    score -= queryFactors.cDotC();

    // Apply OOQ (out-of-quantization) correction
    score -= 0.5f * (docCorr.ooq() + queryFactors.qcDist());

    return score;
}

private int innerProduct(byte[] query, byte[] doc) {
    int sum = 0;
    for (int i = 0; i < query.length; i++) {
        // XOR gives mismatches, NOT gives matches
        int xor = query[i] ^ doc[i];
        sum += 8 - Integer.bitCount(xor & 0xFF);  // Count matching bits
    }
    return sum;
}
```

---

## Key Code References

### Core Quantization
- **OptimizedScalarQuantizer**: `server/src/main/java/org/elasticsearch/index/codec/vectors/OptimizedScalarQuantizer.java:19-219`
- **BinaryQuantizer**: `server/src/main/java/org/elasticsearch/index/codec/vectors/es816/BinaryQuantizer.java:37-385`
- **ESVectorUtil** (quantization utilities): `libs/simdvec/src/main/java/org/elasticsearch/simdvec/ESVectorUtil.java:220-445`

### Indexing (Write Path)
- **ES818BinaryQuantizedVectorsWriter**: `server/src/main/java/org/elasticsearch/index/codec/vectors/es818/ES818BinaryQuantizedVectorsWriter.java:145-219`
- **DiskBBQBulkWriter**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/DiskBBQBulkWriter.java:71-180`
- **ESNextDiskBBQVectorsWriter**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsWriter.java:80-232`

### Searching (Read Path)
- **ES816BinaryFlatVectorsScorer**: `server/src/main/java/org/elasticsearch/index/codec/vectors/es816/ES816BinaryFlatVectorsScorer.java:54-200`
- **ES91OSQVectorsScorer**: `libs/simdvec/src/main/java/org/elasticsearch/simdvec/ES91OSQVectorsScorer.java:51-131`
- **ESNextDiskBBQVectorsReader**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsReader.java:629-793`

### Formats
- **ES814ScalarQuantizedVectorsFormat**: `server/src/main/java/org/elasticsearch/index/codec/vectors/ES814ScalarQuantizedVectorsFormat.java:46-307`
- **ES818BinaryQuantizedVectorsFormat**: `server/src/main/java/org/elasticsearch/index/codec/vectors/es818/ES818BinaryQuantizedVectorsFormat.java`
- **ESNextDiskBBQVectorsFormat**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsFormat.java:76-209`

---

## Summary

1. **What gets quantized:** Document vectors, query vectors, and centroid vectors (but raw centroids also stored in posting lists)
2. **When:** Document vectors at indexing time, query vectors at search time
3. **How:** OSQ uses coordinate descent optimization, Binary uses radial basis quantization
4. **Stored values:** Quantized vectors (1-7 bits) + correction metadata (float32), raw centroids in posting lists
5. **Three main types:** Scalar (OSQ), Binary (RaBitQ), IVF Multi-bit
6. **Scoring:** Bit-level operations + correction factors reconstruct approximate similarities
7. **Compression:** 4.5× to 32× smaller than float32, depending on bit precision
