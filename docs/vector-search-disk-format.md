# Elasticsearch Vector Search: On-Disk Storage Format

## Overview

This document provides a comprehensive guide to how centroids, posting lists, and raw vector values are stored on disk in Elasticsearch's IVF (Inverted File) vector search system.

## Table of Contents
1. [File Structure Overview](#file-structure-overview)
2. [Metadata File Format](#metadata-file-format)
3. [Centroid File Format](#centroid-file-format)
4. [Posting List File Format](#posting-list-file-format)
5. [Doc ID Encoding](#doc-id-encoding)
6. [Data Layout Examples](#data-layout-examples)
7. [Writing Flow](#writing-flow)
8. [Reading Flow](#reading-flow)

---

## File Structure Overview

### Three Main Files Per Field

| File | Extension | Purpose | Writer Class |
|------|-----------|---------|-------------|
| **Metadata** | `.mivf` | Field metadata and offsets | `IVFVectorsWriter` |
| **Centroids** | `.cenivf` | Centroid vectors (quantized + raw) | `ESNextDiskBBQVectorsWriter` |
| **Posting Lists** | `.clivf` | Quantized vectors + doc IDs + corrections | `ESNextDiskBBQVectorsWriter` |

**File Extension Definitions:**
```java
// IVFVectorsWriter.java:79-90
static final String META_EXTENSION = "mivf";        // Metadata
static final String CENTROID_EXTENSION = "cenivf";  // Centroids
static final String CLUSTER_EXTENSION = "clivf";    // Posting lists/clusters
```

---

## Metadata File Format

### Purpose

The `.mivf` file contains high-level information about all vector fields in the segment.

### File Structure

```
┌────────────────────────────────────────┐
│ CODEC HEADER                           │
│   - Magic number                       │
│   - Codec name                         │
│   - Version                            │
│   - Segment ID                         │
│   - Suffix                             │
└────────────────────────────────────────┘
│ FIELD METADATA (repeated)              │
│   ┌────────────────────────────────────┤
│   │ fieldNumber (int)                  │
│   │ rawVectorFormatName (String)       │
│   │ useDirectIOReads (byte) [v2+]      │
│   │ vectorEncoding (int)               │
│   │ similarityFunction (int)           │
│   │ numCentroids (int)                 │
│   │ centroidOffset (long)              │
│   │ centroidLength (long)              │
│   │ postingListOffset (long)           │
│   │ postingListLength (long)           │
│   │ ┌──────────────────────────────────┤
│   │ │ [if centroidLength > 0]          │
│   │ │   globalCentroid (float[dim])    │
│   │ │   globalCentroidDp (int)         │
│   │ └──────────────────────────────────┤
│   │ FORMAT-SPECIFIC METADATA           │
│   │   quantEncodingId (int)            │
│   │   ...                              │
│   └────────────────────────────────────┤
├────────────────────────────────────────┤
│ END MARKER: -1 (int)                   │
├────────────────────────────────────────┤
│ CODEC FOOTER                           │
│   - Algorithm ID                       │
│   - Checksum                           │
└────────────────────────────────────────┘
```

### Writing Metadata

```java
// IVFVectorsWriter.java:269-297
@Override
public void writeMeta(int fieldNumber, int vectorEncoding, IndexOutput out) throws IOException {
    out.writeInt(fieldNumber);
    out.writeString(rawVectorFormat.toString());

    if (version >= VERSION_DIRECT_IO_FLAG) {
        out.writeByte(useDirectIOReads ? (byte) 1 : (byte) 0);
    }

    out.writeVInt(vectorEncoding);
    out.writeVInt(fieldInfo.getVectorSimilarityFunction().ordinal());

    FieldEntry entry = fields.get(fieldNumber);
    out.writeInt(entry.numCentroids);
    out.writeLong(entry.centroidOffset);
    out.writeLong(entry.centroidLength);
    out.writeLong(entry.postingListOffset);
    out.writeLong(entry.postingListLength);

    // Write global centroid if field has vectors
    if (entry.centroidLength > 0) {
        float[] globalCentroid = entry.globalCentroid;
        for (int i = 0; i < globalCentroid.length; i++) {
            out.writeInt(Float.floatToIntBits(globalCentroid[i]));
        }
        out.writeInt(Float.floatToIntBits(entry.globalCentroidDp));
    }

    // Format-specific metadata
    formatWriter.writeMeta(entry, fieldInfo, out);
}
```

### FieldEntry Structure

```java
// IVFVectorsReader.java:586-658
static class FieldEntry {
    final int fieldNumber;
    final VectorEncoding vectorEncoding;
    final VectorSimilarityFunction similarityFunction;
    final int numCentroids;
    final long centroidOffset;
    final long centroidLength;
    final long postingListOffset;
    final long postingListLength;
    final float[] globalCentroid;         // Mean of all vectors
    final float globalCentroidDp;         // Dot product with itself
    final boolean useDirectIO;
    // ... format-specific fields
}
```

---

## Centroid File Format

### Purpose

The `.cenivf` file stores quantized centroid vectors and metadata for locating posting lists.

### Case A: Flat Structure (No Parents)

Used when `numCentroids ≤ centroidsPerParentCluster²`

```
┌────────────────────────────────────────┐
│ CODEC HEADER                           │
└────────────────────────────────────────┘
│ numParents = 0 (VInt)                  │
├────────────────────────────────────────┤
│ QUANTIZED CENTROIDS                    │
│   For each centroid [0..N-1]:          │
│   ┌────────────────────────────────────┤
│   │ quantizedVector (byte[])           │
│   │   - 7-bit quantization             │
│   │   - Size: 7 * (dim+7)/8 bytes      │
│   │ lowerInterval (float)              │
│   │ upperInterval (float)              │
│   │ additionalCorrection (float)       │
│   │ quantizedComponentSum (int)        │
│   └────────────────────────────────────┤
├────────────────────────────────────────┤
│ POSTING LIST METADATA                  │
│   For each centroid [0..N-1]:          │
│   ┌────────────────────────────────────┤
│   │ postingListOffset (long)           │
│   │ postingListLength (long)           │
│   └────────────────────────────────────┘
```

**Writing Code:**
```java
// ESNextDiskBBQVectorsWriter.java:480-502
private void writeCentroidsWithoutParents(
    IndexOutput centroids,
    FieldInfo fieldInfo,
    CentroidSupplier centroidSupplier,
    LongArrayList offsetAndLength
) throws IOException {
    // Indicate no parent structure
    centroids.writeVInt(0);

    // Quantize and write all centroids
    OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer();
    for (int i = 0; i < centroidSupplier.size(); i++) {
        float[] centroid = centroidSupplier.get(i);

        // Quantize to 7 bits
        quantizer.scalarQuantize(
            centroid,
            scratchFloat,
            scratchInt,
            (byte) 7,
            globalCentroid
        );

        // Write quantized vector + corrections
        writeQuantizedValue(centroids, scratchInt, quantizer.getResult());
    }

    // Write posting list offsets and lengths
    for (int i = 0; i < centroidSupplier.size(); i++) {
        centroids.writeLong(offsetAndLength.get(2 * i));      // offset
        centroids.writeLong(offsetAndLength.get(2 * i + 1));  // length
    }
}
```

### Case B: Hierarchical Structure (With Parents)

Used when `numCentroids > centroidsPerParentCluster²`

```
┌────────────────────────────────────────┐
│ CODEC HEADER                           │
└────────────────────────────────────────┘
│ numParents (VInt)                      │
│ maxChildrenPerParent (VInt)            │
├────────────────────────────────────────┤
│ PARENT CENTROIDS (quantized 7-bit)     │
│   For each parent [0..P-1]:            │
│   ┌────────────────────────────────────┤
│   │ quantizedVector (byte[])           │
│   │ lowerInterval (float)              │
│   │ upperInterval (float)              │
│   │ additionalCorrection (float)       │
│   │ quantizedComponentSum (int)        │
│   └────────────────────────────────────┤
├────────────────────────────────────────┤
│ PARENT METADATA                        │
│   For each parent [0..P-1]:            │
│   ┌────────────────────────────────────┤
│   │ childrenOffset (int)               │
│   │   - Index into children array      │
│   │ numChildren (int)                  │
│   │   - Count of children              │
│   └────────────────────────────────────┤
├────────────────────────────────────────┤
│ CHILD CENTROIDS (grouped by parent)    │
│   For parent_0:                        │
│   ┌────────────────────────────────────┤
│   │ child_0 (quantized 7-bit)          │
│   │ child_1 (quantized 7-bit)          │
│   │ ...                                │
│   └────────────────────────────────────┤
│   For parent_1:                        │
│   ┌────────────────────────────────────┤
│   │ child_0 (quantized 7-bit)          │
│   │ ...                                │
│   └────────────────────────────────────┤
├────────────────────────────────────────┤
│ CHILD POSTING LIST METADATA            │
│   For parent_0:                        │
│   ┌────────────────────────────────────┤
│   │ For each child:                    │
│   │   postingListOffset (long)         │
│   │   postingListLength (long)         │
│   └────────────────────────────────────┤
│   For parent_1:                        │
│   ┌────────────────────────────────────┤
│   │ ...                                │
│   └────────────────────────────────────┘
```

**Writing Code:**
```java
// ESNextDiskBBQVectorsWriter.java:435-478
private void writeCentroidsWithParents(
    IndexOutput centroids,
    FieldInfo fieldInfo,
    CentroidGroups centroidGroups,
    LongArrayList offsetAndLength
) throws IOException {
    int numParents = centroidGroups.centroids.length;

    // Write header
    centroids.writeVInt(numParents);
    centroids.writeVInt(centroidGroups.maxVectorsPerCentroidLength);

    // Write quantized parent centroids
    OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer();
    for (int i = 0; i < numParents; i++) {
        quantizer.scalarQuantize(
            centroidGroups.centroids[i],
            scratchFloat,
            scratchInt,
            (byte) 7,
            globalCentroid
        );
        writeQuantizedValue(centroids, scratchInt, quantizer.getResult());
    }

    // Write parent metadata (offset and count for children)
    for (int i = 0; i < numParents; i++) {
        int childOffset = computeChildOffset(centroidGroups.vectors, i);
        centroids.writeInt(childOffset);
        centroids.writeInt(centroidGroups.vectors[i].length);
    }

    // Write quantized child centroids, grouped by parent
    for (int i = 0; i < numParents; i++) {
        int[] childIndices = centroidGroups.vectors[i];
        for (int childIdx : childIndices) {
            float[] childCentroid = centroidSupplier.get(childIdx);
            quantizer.scalarQuantize(
                childCentroid,
                scratchFloat,
                scratchInt,
                (byte) 7,
                globalCentroid
            );
            writeQuantizedValue(centroids, scratchInt, quantizer.getResult());
        }
    }

    // Write posting list offsets/lengths for children
    for (int i = 0; i < numParents; i++) {
        int[] childIndices = centroidGroups.vectors[i];
        for (int childIdx : childIndices) {
            centroids.writeLong(offsetAndLength.get(2 * childIdx));
            centroids.writeLong(offsetAndLength.get(2 * childIdx + 1));
        }
    }
}
```

### Quantized Value Format

Each quantized centroid is stored as:

```java
// ESNextDiskBBQVectorsWriter.java:565-573
private void writeQuantizedValue(
    IndexOutput out,
    int[] quantizedVector,
    OptimizedScalarQuantizer.QuantizationResult result
) throws IOException {
    // Write quantized vector bytes
    idsWriter.writeInts(out, quantizedVector.length, docEncoding, quantizedVector);

    // Write correction metadata
    out.writeInt(Float.floatToIntBits(result.lowerInterval()));
    out.writeInt(Float.floatToIntBits(result.upperInterval()));
    out.writeInt(Float.floatToIntBits(result.additionalCorrection()));
    out.writeInt(result.quantizedComponentSum());
}
```

---

## Posting List File Format

### Purpose

The `.clivf` file stores the actual indexed vectors (quantized) with their document IDs.

### Per-Centroid Posting List Structure

```
┌────────────────────────────────────────┐
│ CODEC HEADER                           │
└────────────────────────────────────────┘
│ POSTING LIST FOR CENTROID 0            │
│   ┌────────────────────────────────────┤
│   │ rawCentroid (float[dimension])     │
│   │   - Full precision centroid        │
│   │   - Size: dim * 4 bytes            │
│   │ centroidDp (int)                   │
│   │   - Float.floatToIntBits(dot(c,c)) │
│   │ vectorCount (VInt)                 │
│   │   - Number of vectors              │
│   │ docIdEncoding (byte)               │
│   │   - Encoding strategy for doc IDs  │
│   │   ┌────────────────────────────────┤
│   │   │ BULK BLOCK 0 (32 vectors)      │
│   │   │   ┌────────────────────────────┤
│   │   │   │ QUANTIZED VECTORS          │
│   │   │   │   vector_0 (byte[])        │
│   │   │   │   vector_1 (byte[])        │
│   │   │   │   ...                      │
│   │   │   │   vector_31 (byte[])       │
│   │   │   ├────────────────────────────┤
│   │   │   │ CORRECTIONS (bulk)         │
│   │   │   │   lowerInterval[32] (float)│
│   │   │   │   upperInterval[32] (float)│
│   │   │   │   compSum[32] (short)      │
│   │   │   │   addCorr[32] (float)      │
│   │   │   ├────────────────────────────┤
│   │   │   │ DOC IDS                    │
│   │   │   │   (encoded as deltas)      │
│   │   │   └────────────────────────────┤
│   │   │ BULK BLOCK 1...                │
│   │   │ ...                            │
│   │   │ TAIL BLOCK (< 32 vectors)      │
│   │   │   (individual scoring)         │
│   │   └────────────────────────────────┤
│   └────────────────────────────────────┤
│ POSTING LIST FOR CENTROID 1...         │
│ ...                                    │
└────────────────────────────────────────┘
```

### Writing Posting Lists

```java
// ESNextDiskBBQVectorsWriter.java:80-180
private List<ClusterInfo> buildAndWritePostingsLists(
    IndexOutput clustersOutput,
    List<ClusterBuffer> clusters,
    FieldInfo fieldInfo,
    OptimizedScalarQuantizer quantizer,
    float[] globalCentroid
) throws IOException {
    List<ClusterInfo> clusterInfos = new ArrayList<>(clusters.size());

    for (ClusterBuffer cluster : clusters) {
        long startOffset = clustersOutput.getFilePointer();

        // Write raw centroid (full precision)
        float[] rawCentroid = cluster.centroid();
        for (int i = 0; i < rawCentroid.length; i++) {
            clustersOutput.writeInt(Float.floatToIntBits(rawCentroid[i]));
        }

        // Write centroid dot product
        float centroidDp = VectorUtil.dotProduct(rawCentroid, rawCentroid);
        clustersOutput.writeInt(Float.floatToIntBits(centroidDp));

        // Write vector count
        int vectorCount = cluster.size();
        clustersOutput.writeVInt(vectorCount);

        // Determine and write doc ID encoding
        byte docEncoding = determineDocIdEncoding(cluster.docIds());
        clustersOutput.writeByte(docEncoding);

        // Write vectors in bulk blocks (32 at a time)
        DiskBBQBulkWriter bulkWriter = createBulkWriter(
            fieldInfo,
            clustersOutput,
            quantizer,
            globalCentroid,
            rawCentroid,
            docEncoding
        );

        int limit = vectorCount - BULK_SIZE + 1;
        for (int i = 0; i < limit; i += BULK_SIZE) {
            bulkWriter.writeBulk(
                cluster.vectors(),
                cluster.docIds(),
                i,
                BULK_SIZE
            );
        }

        // Write tail (remaining < BULK_SIZE vectors)
        if (limit < vectorCount) {
            int remaining = vectorCount - limit;
            for (int i = limit; i < vectorCount; i++) {
                bulkWriter.writeIndividual(
                    cluster.vectors()[i],
                    cluster.docIds()[i]
                );
            }
        }

        long endOffset = clustersOutput.getFilePointer();
        clusterInfos.add(new ClusterInfo(
            cluster.centroidOrdinal(),
            startOffset,
            endOffset - startOffset
        ));
    }

    return clusterInfos;
}
```

### Bulk Writing

```java
// DiskBBQBulkWriter.java:71-180
public void writeBulk(
    float[][] vectors,
    int[] docIds,
    int offset,
    int count
) throws IOException {
    assert count <= BULK_SIZE;

    // Quantize all vectors in bulk
    for (int i = 0; i < count; i++) {
        float[] vector = vectors[offset + i];

        // Center relative to centroid
        ESVectorUtil.subtract(vector, centroid, scratchVector);

        // Quantize
        OptimizedScalarQuantizer.QuantizationResult result = quantizer.scalarQuantize(
            scratchVector,
            residual,
            quantizedScratch,
            bits,
            null  // Already centered
        );

        // Store quantized vector
        idsWriter.writeInts(
            output,
            dimension,
            quantEncoding,
            quantizedScratch
        );

        // Store corrections for later bulk write
        correctionsLower[i] = result.lowerInterval();
        correctionsUpper[i] = result.upperInterval();
        correctionsSum[i] = (short) result.quantizedComponentSum();
        correctionsAdd[i] = result.additionalCorrection();
    }

    // Write all corrections in bulk
    for (int i = 0; i < count; i++) {
        output.writeInt(Float.floatToIntBits(correctionsLower[i]));
    }
    for (int i = 0; i < count; i++) {
        output.writeInt(Float.floatToIntBits(correctionsUpper[i]));
    }
    for (int i = 0; i < count; i++) {
        output.writeShort(correctionsSum[i]);
    }
    for (int i = 0; i < count; i++) {
        output.writeInt(Float.floatToIntBits(correctionsAdd[i]));
    }

    // Write doc IDs (encoded as deltas)
    docIdsWriter.writeDocIds(output, docIds, offset, count);
}
```

---

## Doc ID Encoding

### Encoding Strategies

Doc IDs are stored as deltas (difference from previous ID) using various encoding schemes:

```java
// DocIdsWriter.java:54-91
public static final byte CONTINUOUS_IDS = -2;    // Sequential IDs: [n, n+1, n+2, ...]
public static final byte DELTA_BPV_16 = 16;      // 16-bit deltas (max gap 65535)
public static final byte BPV_21 = 21;            // 21-bit packed values
public static final byte BPV_24 = 24;            // 24-bit packed values
public static final byte BPV_32 = 32;            // 32-bit full values
```

### CONTINUOUS_IDS Format

For sequential doc IDs:

```
┌────────────────────────────────┐
│ encoding = -2 (byte)           │
│ startDocId (VInt)              │
└────────────────────────────────┘
```

Reconstruction: `docId[i] = startDocId + i`

### DELTA_BPV_16 Format

For doc IDs with small gaps:

```
┌────────────────────────────────┐
│ encoding = 16 (byte)           │
│ minDocId (VInt)                │
│ packedDeltas[count/2] (int)    │
│   - 2 deltas per int           │
│   - delta0: bits [31:16]       │
│   - delta1: bits [15:0]        │
│ [if odd count]                 │
│   lastDelta (short)            │
└────────────────────────────────┘
```

**Writing Code:**
```java
// DocIdsWriter.java:133-157
private void writeDeltaBPV16(IndexOutput out, int[] docIds, int offset, int count) throws IOException {
    int minDocId = docIds[offset];
    out.writeVInt(minDocId);

    int prevDocId = minDocId;
    int half = count / 2;

    // Pack 2 deltas per int
    for (int i = 0; i < half; i++) {
        int docId0 = docIds[offset + 2 * i];
        int docId1 = docIds[offset + 2 * i + 1];

        int delta0 = docId0 - prevDocId;
        int delta1 = docId1 - docId0;

        int combined = (delta0 << 16) | (delta1 & 0xFFFF);
        out.writeInt(combined);

        prevDocId = docId1;
    }

    // Write last delta if count is odd
    if (count % 2 == 1) {
        int lastDocId = docIds[offset + count - 1];
        int lastDelta = lastDocId - prevDocId;
        out.writeShort((short) lastDelta);
    }
}
```

### BPV_24 Format

For wider doc ID ranges:

```
┌────────────────────────────────┐
│ encoding = 24 (byte)           │
│ packedValues[count * 3 / 4]    │
│   - 4 values per 3 ints        │
│   - Each value: 24 bits        │
└────────────────────────────────┘
```

### BPV_32 Format

For full 32-bit doc IDs:

```
┌────────────────────────────────┐
│ encoding = 32 (byte)           │
│ docIds[count] (int)            │
│   - Raw 32-bit doc IDs         │
└────────────────────────────────┘
```

### Determining Encoding

```java
// DocIdsWriter.java:93-112
public byte determineEncoding(int[] docIds, int offset, int count) {
    if (count == 0) return CONTINUOUS_IDS;

    int minDocId = docIds[offset];
    int maxDocId = docIds[offset + count - 1];

    // Check if continuous
    if (maxDocId - minDocId == count - 1) {
        return CONTINUOUS_IDS;
    }

    // Check maximum delta
    int maxDelta = 0;
    for (int i = 1; i < count; i++) {
        int delta = docIds[offset + i] - docIds[offset + i - 1];
        if (delta > maxDelta) {
            maxDelta = delta;
        }
    }

    if (maxDelta <= 65535) return DELTA_BPV_16;
    if (maxDocId <= 0xFFFFFF) return BPV_24;
    return BPV_32;
}
```

---

## Data Layout Examples

### Example: 768-Dimensional Field, 1-Bit Encoding

**Configuration:**
- Dimension: 768
- Encoding: 1-bit (ONE_BIT_4BIT_QUERY)
- Vectors: 100,000
- Centroids: 256

**Size Calculations:**

**Metadata File (.mivf):**
```
Header: ~50 bytes
Field entry: ~120 bytes
Footer: ~16 bytes
Total: ~200 bytes
```

**Centroids File (.cenivf):**
```
Header: 16 bytes
Centroids (256):
  - Quantized (7-bit): 256 * (7 * 96) = 172,032 bytes
  - Corrections: 256 * 16 = 4,096 bytes
Posting offsets/lengths: 256 * 16 = 4,096 bytes
Total: ~180 KB
```

**Posting Lists File (.clivf):**
```
Per centroid header:
  - Raw centroid: 768 * 4 = 3,072 bytes
  - Centroid DP: 4 bytes
  - Vector count: ~2 bytes
  - Doc encoding: 1 byte

Per vector (avg ~390 per centroid):
  - Quantized: (768 * 1 bit) / 8 = 96 bytes
  - Corrections: 14 bytes (4+4+2+4)
  - Doc ID: ~2 bytes (delta encoding)

Total per centroid: ~43 KB
Total for 256 centroids: ~11 MB
```

### Example: 1024-Dimensional Field, 4-Bit Encoding

**Configuration:**
- Dimension: 1024
- Encoding: 4-bit (FOUR_BIT_SYMMETRIC)
- Vectors: 1,000,000
- Centroids: 4096

**Size Calculations:**

**Centroids File:**
```
Hierarchical (4096 > 256²):
  - Parent centroids: 64 * (4 * 128 + 16) = ~33 KB
  - Child centroids: 4096 * (4 * 128 + 16) = ~2.1 MB
  - Offsets/lengths: 4096 * 16 = ~64 KB
Total: ~2.2 MB
```

**Posting Lists File:**
```
Per vector:
  - Quantized: (1024 * 4 bits) / 8 = 512 bytes
  - Corrections: 14 bytes
  - Doc ID: ~3 bytes

Per centroid (avg ~244 vectors):
  - Header: ~4.1 KB
  - Vectors: 244 * 526 = ~125 KB

Total: ~520 MB
```

---

## Writing Flow

### Complete Write Sequence

```java
// ESNextDiskBBQVectorsWriter.java:138-232
@Override
public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    // 1. Cluster vectors into centroids
    KMeansResult clusterResult = HierarchicalKMeans.cluster(
        vectorValues,
        targetClusterSize
    );

    // 2. Build cluster buffers (group vectors by centroid)
    List<ClusterBuffer> clusters = buildClusterBuffers(
        vectorValues,
        clusterResult
    );

    // 3. Sort vectors by doc ID within each cluster
    for (ClusterBuffer cluster : clusters) {
        sortByDocId(cluster);
    }

    // 4. Write posting lists to .clivf file
    List<ClusterInfo> clusterInfos = buildAndWritePostingsLists(
        clustersOutput,
        clusters,
        fieldInfo,
        quantizer,
        globalCentroid
    );

    // 5. Extract offsets and lengths
    LongArrayList offsetsAndLengths = new LongArrayList(clusterInfos.size() * 2);
    for (ClusterInfo info : clusterInfos) {
        offsetsAndLengths.add(info.offset());
        offsetsAndLengths.add(info.length());
    }

    // 6. Write centroids to .cenivf file
    writeCentroids(
        fieldInfo,
        centroidsOutput,
        new CentroidSupplier(clusterResult.centroids()),
        offsetsAndLengths
    );

    // 7. Update field entry with offsets
    FieldEntry entry = new FieldEntry(
        fieldInfo.number,
        clusterResult.centroids().length,
        centroidOffset,
        centroidLength,
        postingListOffset,
        postingListLength,
        globalCentroid,
        globalCentroidDp
    );
    fields.put(fieldInfo.number, entry);
}
```

---

## Reading Flow

### Complete Read Sequence

```java
// ESNextDiskBBQVectorsReader.java:113-194
@Override
public CentroidIterator getCentroidIterator(...) throws IOException {
    // 1. Read centroid file header
    int numParents = centroids.readVInt();

    // 2. Quantize query
    byte[] quantized = new byte[dimension];
    OptimizedScalarQuantizer.QuantizationResult queryParams = quantizer.scalarQuantize(
        targetQuery,
        scratchFloat,
        scratchInt,
        (byte) 7,
        globalCentroid
    );

    // 3. Score centroids
    CentroidIterator iterator;
    if (numParents > 0) {
        // Hierarchical: score parents, then children
        iterator = getCentroidIteratorWithParents(...);
    } else {
        // Flat: score all centroids
        iterator = getCentroidIteratorNoParent(...);
    }

    // 4. Return prefetching iterator
    return getPostingListPrefetchIterator(iterator, postingListSlice);
}
```

### Reading Posting Lists

```java
// ESNextDiskBBQVectorsReader.java:629-688 (MemorySegmentPostingsVisitor)
@Override
public long resetPostingsScorer(long offset) throws IOException {
    // Seek to posting list
    indexInput.seek(offset);
    slicePos = offset;

    // Read raw centroid
    for (int i = 0; i < dimension; i++) {
        centroid[i] = Float.intBitsToFloat(indexInput.readInt());
    }

    // Read centroid dot product
    centroidDp = Float.intBitsToFloat(indexInput.readInt());

    // Read vector count
    vectors = indexInput.readVInt();

    // Read doc ID encoding
    docEncoding = indexInput.readByte();

    return vectors;
}
```

### Scoring Vectors

```java
// ESNextDiskBBQVectorsReader.java:721-793 (MemorySegmentPostingsVisitor)
@Override
public int visit(KnnCollector knnCollector) throws IOException {
    indexInput.seek(slicePos);
    int scoredDocs = 0;

    // Process full batches (16 vectors)
    int limit = vectors - BULK_SIZE + 1;
    for (int i = 0; i < limit; i += BULK_SIZE) {
        // Read doc IDs
        readDocIds(BULK_SIZE);

        // Quantize query if needed
        quantizeQueryIfNecessary();

        // Bulk score using SIMD
        float maxScore = osqVectorsScorer.scoreBulk(
            quantizedQueryScratch,
            queryCorrections.lowerInterval(),
            queryCorrections.upperInterval(),
            queryCorrections.quantizedComponentSum(),
            queryCorrections.additionalCorrection(),
            fieldInfo.getVectorSimilarityFunction(),
            centroidDp,
            scores
        );

        // Collect results
        if (knnCollector.minCompetitiveSimilarity() < maxScore) {
            collectBulk(knnCollector, scores);
        }

        scoredDocs += BULK_SIZE;
    }

    // Handle tail (< 16 vectors)
    // ...

    return scoredDocs;
}
```

---

## Key Code References

### File Format
- **File extensions**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/IVFVectorsWriter.java:79-90`
- **Metadata writing**: Lines 269-297
- **FieldEntry**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/IVFVectorsReader.java:586-658`

### Centroids
- **Writing with parents**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsWriter.java:435-478`
- **Writing without parents**: Lines 480-502
- **Reading**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsReader.java:113-194`

### Posting Lists
- **Writing**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsWriter.java:80-180`
- **Bulk writing**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/DiskBBQBulkWriter.java:71-180`
- **Reading**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsReader.java:629-793`

### Doc ID Encoding
- **DocIdsWriter**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/DocIdsWriter.java:54-187`
- **Encoding determination**: Lines 93-112
- **Reading**: Lines 189-255

---

## Summary

1. **Three files per field**: metadata (`.mivf`), centroids (`.cenivf`), posting lists (`.clivf`)
2. **Centroid storage varies**: flat structure for small counts, hierarchical for large counts
3. **Vectors stored quantized**: 1-7 bits per dimension with correction metadata
4. **Doc IDs delta-encoded**: multiple encoding strategies based on ID distribution
5. **Bulk operations**: 32 vectors written/read at once for efficiency
6. **File offsets tracked**: metadata file contains offsets into centroid and posting list files
7. **Raw centroids in posting lists**: full-precision centroids stored with each posting list for accurate scoring
