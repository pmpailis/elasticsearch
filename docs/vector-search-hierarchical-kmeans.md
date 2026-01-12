# Hierarchical K-Means in Elasticsearch Vector Search

## Overview

This document explains the hierarchical k-means clustering algorithm used in Elasticsearch's IVF (Inverted File) vector search system, including how parent/child centroid relationships are established and used during indexing and search.

## Table of Contents
1. [Hierarchical K-Means Algorithm](#hierarchical-k-means-algorithm)
2. [Parent/No-Parent References](#parentno-parent-references)
3. [Clustering Implementation](#clustering-implementation)
4. [Search-Time Traversal](#search-time-traversal)
5. [Key Code References](#key-code-references)

---

## Hierarchical K-Means Algorithm

### Core Algorithm

**Primary Implementation:**
- **File:** `server/src/main/java/org/elasticsearch/index/codec/vectors/cluster/HierarchicalKMeans.java`
- **Lines:** 20-303

The hierarchical k-means algorithm recursively partitions the vector space to create better clusters than naive k-means. It uses Lloyd's algorithm iteratively with splitting and merging strategies.

### Key Parameters

```java
// HierarchicalKMeans.java:24-28
private static final int MAXK = 128;                        // Maximum clusters allowed
private static final int MAX_ITERATIONS_DEFAULT = 6;        // Lloyd's algorithm iterations
private static final int SAMPLES_PER_CLUSTER_DEFAULT = 64;  // Sample size per cluster
private static final float DEFAULT_SOAR_LAMBDA = 1.0f;      // SOAR algorithm lambda
private static final int NO_SOAR_ASSIGNMENT = -1;           // No SOAR assignment sentinel
```

### Main Entry Point

```java
// HierarchicalKMeans.java:114-144
public static KMeansResult cluster(FloatVectorValues vectors, int targetSize) throws IOException {
    // Base case: if vectors fit in single cluster
    if (vectors.size() <= targetSize) {
        float[] centroid = new float[vectors.dimension()];
        // Calculate mean of all vectors
        for (int i = 0; i < vectors.size(); i++) {
            ESVectorUtil.add(centroid, vectors.vectorValue(i));
        }
        ESVectorUtil.divInPlace(centroid, vectors.size());

        int[] assignments = new int[vectors.size()];
        Arrays.fill(assignments, 0);

        return new KMeansResult(
            new KMeansIntermediate(new float[][] { centroid }, assignments, null),
            null  // No SOAR assignments for single cluster
        );
    }

    // Recursive case: cluster and split
    KMeansIntermediate result = clusterAndSplit(vectors, targetSize);

    // Refine with neighborhood-aware clustering if multiple centroids
    if (result.centroids().length > 1 && result.centroids().length < vectors.size()) {
        return KMeansLocal.cluster(vectors, result, DEFAULT_SOAR_LAMBDA);
    }

    return new KMeansResult(result, null);
}
```

### Recursive Splitting Strategy

The algorithm recursively splits clusters that are too large:

```java
// HierarchicalKMeans.java:146-226
private static KMeansIntermediate clusterAndSplit(
    FloatVectorValues vectors,
    int targetSize
) throws IOException {
    // Calculate initial k based on target
    int k = Math.min(MAXK, Math.max(2, (int) Math.ceil((double) vectors.size() / targetSize)));

    // Run Lloyd's algorithm for k clusters
    KMeansIntermediate result = KMeans.cluster(
        vectors,
        k,
        new ProgressListener() {},
        MAX_ITERATIONS_DEFAULT,
        SAMPLES_PER_CLUSTER_DEFAULT,
        KMeans.INERTIA_CHECK
    );

    // Count vectors per cluster
    int[] clusterCounts = new int[k];
    for (int assignment : result.assignments()) {
        clusterCounts[assignment]++;
    }

    // Recursively split oversized clusters
    for (int c = 0; c < k; c++) {
        int count = clusterCounts[c];

        if (count > (int) (1.34 * targetSize)) {
            // Cluster too large - split it
            FloatVectorValues clusterVectors = extractClusterVectors(vectors, result.assignments(), c);
            KMeansIntermediate subResult = clusterAndSplit(clusterVectors, targetSize);

            // Merge results - replace parent centroid with children
            result = updateAssignmentsWithRecursiveSplit(result, c, subResult);

        } else if (count == 0) {
            // Empty cluster - remove it
            result = removeEmptyCluster(result, c);
        }
    }

    return result;
}
```

### Centroid Remapping During Recursion

When a parent cluster splits into N children, all centroid ordinals must be updated:

```java
// HierarchicalKMeans.java:260-302
private static KMeansIntermediate updateAssignmentsWithRecursiveSplit(
    KMeansIntermediate result,
    int clusterToSplit,
    KMeansIntermediate recursiveSplit
) {
    int numOldCentroids = result.centroids().length;
    int numNewCentroids = recursiveSplit.centroids().length;
    int numTotalCentroids = numOldCentroids - 1 + numNewCentroids;

    // Build new centroid array: [0...clusterToSplit-1, newCentroids..., clusterToSplit+1...end]
    float[][] newCentroids = new float[numTotalCentroids][];

    // Copy centroids before split point
    System.arraycopy(result.centroids(), 0, newCentroids, 0, clusterToSplit);

    // Insert new child centroids
    System.arraycopy(
        recursiveSplit.centroids(),
        0,
        newCentroids,
        clusterToSplit,
        numNewCentroids
    );

    // Copy centroids after split point
    System.arraycopy(
        result.centroids(),
        clusterToSplit + 1,
        newCentroids,
        clusterToSplit + numNewCentroids,
        numOldCentroids - clusterToSplit - 1
    );

    // Update assignments
    int[] newAssignments = new int[result.assignments().length];
    for (int i = 0; i < result.assignments().length; i++) {
        int oldAssignment = result.assignments()[i];

        if (oldAssignment == clusterToSplit) {
            // This vector belongs to split cluster - use recursive assignment
            int localIndex = getLocalIndex(result.assignments(), i, clusterToSplit);
            newAssignments[i] = clusterToSplit + recursiveSplit.assignments()[localIndex];
        } else if (oldAssignment > clusterToSplit) {
            // This vector's centroid moved - adjust ordinal
            newAssignments[i] = oldAssignment - 1 + numNewCentroids;
        } else {
            // This vector's centroid unchanged
            newAssignments[i] = oldAssignment;
        }
    }

    return new KMeansIntermediate(newCentroids, newAssignments, null);
}
```

**Key Insight:** The hierarchical tree is flattened into a linear centroid array. When a parent splits, its children are inserted into the array and all subsequent ordinals are shifted.

---

## Parent/No-Parent References

### Decision Point: When to Use Hierarchy

The system determines whether to use a two-level hierarchy based on the **total number of centroids**:

```java
// ESNextDiskBBQVectorsWriter.java:399-420
private void writeCentroids(
    FieldInfo fieldInfo,
    IndexOutput centroids,
    CentroidSupplier centroidSupplier,
    LongArrayList offsetAndLength
) throws IOException {
    int centroidsPerParentCluster = fieldInfo.getAttribute(CENTROIDS_PER_PARENT_CLUSTER);

    // DECISION: Use hierarchy if numCentroids > centroidsPerParentCluster²
    if (centroidSupplier.size() > centroidsPerParentCluster * centroidsPerParentCluster) {
        // WITH PARENTS: Two-level hierarchy
        CentroidGroups centroidGroups = buildCentroidGroups(
            centroidSupplier,
            centroidsPerParentCluster,
            fieldInfo.getVectorDimension(),
            fieldInfo.getVectorSimilarityFunction()
        );
        writeCentroidsWithParents(centroids, fieldInfo, centroidGroups, offsetAndLength);
    } else {
        // NO PARENTS: Flat structure
        writeCentroidsWithoutParents(centroids, fieldInfo, centroidSupplier, offsetAndLength);
    }
}
```

**Example:**
- If `centroidsPerParentCluster = 256`:
  - `numCentroids ≤ 65,536` → Flat structure (no parents)
  - `numCentroids > 65,536` → Hierarchical structure (with parents)

### Building Centroid Groups

When using hierarchy, centroids are clustered into parent groups:

```java
// ESNextDiskBBQVectorsWriter.java:506-533
private static CentroidGroups buildCentroidGroups(
    CentroidSupplier centroidSupplier,
    int centroidsPerParentCluster,
    int dimension,
    VectorSimilarityFunction similarityFunction
) throws IOException {
    // Run hierarchical k-means on the centroids themselves
    KMeansResult parentClusterResult = HierarchicalKMeans.cluster(
        new CentroidVectorValues(centroidSupplier, dimension),
        centroidsPerParentCluster
    );

    // Group child centroids by parent assignment
    int numParents = parentClusterResult.centroids().length;
    List<List<Integer>> groups = new ArrayList<>(numParents);
    for (int i = 0; i < numParents; i++) {
        groups.add(new ArrayList<>());
    }

    for (int i = 0; i < centroidSupplier.size(); i++) {
        int parentAssignment = parentClusterResult.assignments()[i];
        groups.get(parentAssignment).add(i);
    }

    // Convert to array structure
    int[][] childIndices = new int[numParents][];
    for (int i = 0; i < numParents; i++) {
        childIndices[i] = groups.get(i).stream().mapToInt(Integer::intValue).toArray();
    }

    return new CentroidGroups(
        parentClusterResult.centroids(),
        childIndices,
        maxChildrenPerParent
    );
}
```

**Structure:**
```
CentroidGroups record:
├─ centroids: float[][]          // Parent centroid vectors
├─ vectors: int[][]              // Child centroid indices per parent
└─ maxVectorsPerCentroidLength   // Max children in any parent group
```

---

## Writing Centroids to Disk

### With Parents (Hierarchical)

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
    centroids.writeVInt(numParents);                              // Number of parent centroids
    centroids.writeVInt(centroidGroups.maxVectorsPerCentroidLength);  // Max children per parent

    // Write parent centroids (7-bit quantized)
    OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer();
    for (int i = 0; i < numParents; i++) {
        float[] parentCentroid = centroidGroups.centroids[i];
        quantizer.scalarQuantize(parentCentroid, scratchFloat, scratchInt, (byte) 7, globalCentroid);
        writeQuantizedValue(centroids, scratchInt, quantizer.getResult());
    }

    // Write parent metadata (offset and count of children)
    for (int i = 0; i < numParents; i++) {
        int[] children = centroidGroups.vectors[i];
        int childOffset = computeOffset(centroidGroups.vectors, i);
        centroids.writeInt(childOffset);    // Starting index in children array
        centroids.writeInt(children.length); // Number of children
    }

    // Write child centroids (7-bit quantized), grouped by parent
    for (int i = 0; i < numParents; i++) {
        int[] childIndices = centroidGroups.vectors[i];
        for (int childIdx : childIndices) {
            float[] childCentroid = centroidSupplier.get(childIdx);
            quantizer.scalarQuantize(childCentroid, scratchFloat, scratchInt, (byte) 7, globalCentroid);
            writeQuantizedValue(centroids, scratchInt, quantizer.getResult());
        }
    }

    // Write posting list offsets and lengths for each child centroid
    for (int i = 0; i < numParents; i++) {
        int[] childIndices = centroidGroups.vectors[i];
        for (int childIdx : childIndices) {
            centroids.writeLong(offsetAndLength.get(2 * childIdx));      // offset
            centroids.writeLong(offsetAndLength.get(2 * childIdx + 1));  // length
        }
    }
}
```

**File Format:**
```
┌─────────────────────────────────────┐
│ numParents (VInt)                   │
│ maxChildrenPerParent (VInt)         │
├─────────────────────────────────────┤
│ Parent Centroids (quantized 7-bit)  │
│   [parent_0] ... [parent_N-1]       │
├─────────────────────────────────────┤
│ Parent Metadata                     │
│   For each parent:                  │
│     childOffset (Int)               │
│     numChildren (Int)               │
├─────────────────────────────────────┤
│ Child Centroids (grouped by parent) │
│   [children of parent_0]            │
│   [children of parent_1]            │
│   ...                               │
├─────────────────────────────────────┤
│ Posting List Offsets/Lengths        │
│   For each child (grouped by parent)│
│     offset (Long)                   │
│     length (Long)                   │
└─────────────────────────────────────┘
```

### Without Parents (Flat)

```java
// ESNextDiskBBQVectorsWriter.java:480-502
private void writeCentroidsWithoutParents(
    IndexOutput centroids,
    FieldInfo fieldInfo,
    CentroidSupplier centroidSupplier,
    LongArrayList offsetAndLength
) throws IOException {
    // Write header indicating no parents
    centroids.writeVInt(0);  // numParents = 0

    // Write all centroids (7-bit quantized)
    OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer();
    for (int i = 0; i < centroidSupplier.size(); i++) {
        float[] centroid = centroidSupplier.get(i);
        quantizer.scalarQuantize(centroid, scratchFloat, scratchInt, (byte) 7, globalCentroid);
        writeQuantizedValue(centroids, scratchInt, quantizer.getResult());
    }

    // Write posting list offsets and lengths
    for (int i = 0; i < centroidSupplier.size(); i++) {
        centroids.writeLong(offsetAndLength.get(2 * i));      // offset
        centroids.writeLong(offsetAndLength.get(2 * i + 1));  // length
    }
}
```

**File Format:**
```
┌─────────────────────────────────────┐
│ 0 (VInt) - indicates no parents     │
├─────────────────────────────────────┤
│ All Centroids (quantized 7-bit)     │
│   [centroid_0] ... [centroid_N-1]   │
├─────────────────────────────────────┤
│ Posting List Offsets/Lengths        │
│   For each centroid:                │
│     offset (Long)                   │
│     length (Long)                   │
└─────────────────────────────────────┘
```

---

## Search-Time Traversal

### Reading Centroids

```java
// ESNextDiskBBQVectorsReader.java:112-193
@Override
public CentroidIterator getCentroidIterator(
    FieldInfo fieldInfo,
    int numCentroids,
    IndexInput centroids,
    float[] targetQuery,
    IndexInput postingListSlice,
    AcceptDocs acceptDocs,
    float approximateCost,
    FloatVectorValues values,
    float visitRatio
) throws IOException {
    // ... (filtering and quantization) ...

    // Read parent count to determine structure
    int numParents = centroids.readVInt();

    CentroidIterator centroidIterator;
    if (numParents > 0) {
        // WITH PARENTS: Two-level hierarchy search
        centroidIterator = getCentroidIteratorWithParents(
            fieldInfo,
            centroids,
            numParents,
            numCentroids,
            scorer,
            quantized,
            queryParams,
            globalCentroidDp,
            visitRatio * centroidOversampling,  // Oversample for recall
            acceptCentroids
        );
    } else {
        // NO PARENTS: Flat search
        centroidIterator = getCentroidIteratorNoParent(
            fieldInfo,
            centroids,
            numCentroids,
            scorer,
            quantized,
            queryParams,
            globalCentroidDp,
            acceptCentroids
        );
    }

    return getPostingListPrefetchIterator(centroidIterator, postingListSlice);
}
```

### Flat Search (No Parents)

Simple: score all centroids and return them in order:

```java
// ESNextDiskBBQVectorsReader.java:266-308
private static CentroidIterator getCentroidIteratorNoParent(
    FieldInfo fieldInfo,
    IndexInput centroids,
    int numCentroids,
    ES92Int7VectorsScorer scorer,
    byte[] quantizeQuery,
    OptimizedScalarQuantizer.QuantizationResult queryParams,
    float globalCentroidDp,
    FixedBitSet acceptCentroids
) throws IOException {
    // Create priority queue for all centroids
    final NeighborQueue neighborQueue = new NeighborQueue(numCentroids, true);

    // Score all centroids in bulk
    final long centroidQuantizeSize = fieldInfo.getVectorDimension() + 3 * Float.BYTES + Integer.BYTES;
    score(
        neighborQueue,
        numCentroids,
        0,  // offset = 0
        scorer,
        centroids,
        centroidQuantizeSize,
        quantizeQuery,
        queryParams,
        globalCentroidDp,
        fieldInfo.getVectorSimilarityFunction(),
        new float[ES92Int7VectorsScorer.BULK_SIZE],
        acceptCentroids
    );

    // Return iterator over scored centroids
    return new CentroidIterator() {
        @Override
        public boolean hasNext() {
            return neighborQueue.size() > 0;
        }

        @Override
        public CentroidMeta nextCentroidMeta() throws IOException {
            int centroidOrd = neighborQueue.pop();
            long offset = readOffsetForCentroid(centroids, centroidOrd);
            return new CentroidMeta(centroidOrd, offset);
        }
    };
}
```

### Hierarchical Search (With Parents)

Two-stage scoring: parents first, then lazily score children:

```java
// ESNextDiskBBQVectorsReader.java:310-441
private static CentroidIterator getCentroidIteratorWithParents(
    FieldInfo fieldInfo,
    IndexInput centroids,
    int numParents,
    int numCentroids,
    ES92Int7VectorsScorer scorer,
    byte[] quantizeQuery,
    OptimizedScalarQuantizer.QuantizationResult queryParams,
    float globalCentroidDp,
    float visitRatio,
    FixedBitSet acceptCentroids
) throws IOException {
    int numToVisit = Math.min(numCentroids, (int) Math.ceil(numCentroids * visitRatio));

    // Three queues for managing traversal
    final NeighborQueue parentsQueue = new NeighborQueue(numToVisit, true);
    final NeighborQueue currentParentQueue = new NeighborQueue(numToVisit, true);
    final NeighborQueue neighborQueue = new NeighborQueue(numToVisit, true);

    // Read parent metadata
    int maxChildrenPerParent = centroids.readVInt();
    long parentMetadataOffset = centroids.getFilePointer() +
        (long) numParents * (fieldInfo.getVectorDimension() + 3 * Float.BYTES + Integer.BYTES);

    // Score all parent centroids
    long centroidQuantizeSize = fieldInfo.getVectorDimension() + 3 * Float.BYTES + Integer.BYTES;
    score(
        parentsQueue,
        numParents,
        0,
        scorer,
        centroids,
        centroidQuantizeSize,
        quantizeQuery,
        queryParams,
        globalCentroidDp,
        fieldInfo.getVectorSimilarityFunction(),
        new float[ES92Int7VectorsScorer.BULK_SIZE],
        null  // Don't filter parents
    );

    // Calculate child metadata offset
    long childrenOffset = parentMetadataOffset + (2L * Integer.BYTES * numParents);

    // Populate initial children from best parent
    if (parentsQueue.size() > 0) {
        int parentOrd = parentsQueue.pop();
        populateChildren(
            centroids,
            parentMetadataOffset,
            childrenOffset,
            parentOrd,
            currentParentQueue,
            neighborQueue,
            scorer,
            quantizeQuery,
            queryParams,
            globalCentroidDp,
            fieldInfo,
            acceptCentroids
        );
    }

    // Return iterator that lazily loads more children as needed
    return new CentroidIterator() {
        private int currentParent = -1;

        @Override
        public boolean hasNext() {
            return neighborQueue.size() > 0 || currentParentQueue.size() > 0 || parentsQueue.size() > 0;
        }

        @Override
        public CentroidMeta nextCentroidMeta() throws IOException {
            // If current parent exhausted, load next parent's children
            if (neighborQueue.size() == 0 && currentParentQueue.size() == 0 && parentsQueue.size() > 0) {
                currentParent = parentsQueue.pop();
                populateChildren(
                    centroids,
                    parentMetadataOffset,
                    childrenOffset,
                    currentParent,
                    currentParentQueue,
                    neighborQueue,
                    scorer,
                    quantizeQuery,
                    queryParams,
                    globalCentroidDp,
                    fieldInfo,
                    acceptCentroids
                );
            }

            // Move child from currentParentQueue to neighborQueue if needed
            if (neighborQueue.size() == 0 && currentParentQueue.size() > 0) {
                int childOrd = currentParentQueue.pop();
                neighborQueue.add(childOrd, 0);  // Already scored
            }

            // Return next child centroid
            int centroidOrd = neighborQueue.pop();
            long offset = readChildOffsetForCentroid(centroids, childrenOffset, centroidOrd);
            return new CentroidMeta(centroidOrd, offset);
        }
    };
}
```

**Search Strategy:**
```
1. Score ALL parent centroids (e.g., 256 parents)
2. Add to parentsQueue (priority queue by score)
3. Pop best parent
4. Score its children (e.g., 256 children)
5. Add to neighborQueue
6. Yield children in score order
7. When children exhausted, pop next parent from parentsQueue
8. Repeat until neighborQueue reaches target size
```

**Benefits:**
- Only score children of relevant parents
- Parents act as routing mechanism
- Reduces centroid scoring from O(N) to O(√N + k·√N)

---

## Neighborhood Awareness

### NeighborHood Structure

```java
// NeighborHood.java:32
public record NeighborHood(int[] neighbors, float maxIntraDistance) {
    // neighbors: ordinals of nearby centroids
    // maxIntraDistance: threshold for well-represented vectors
}
```

### Computing Neighborhoods

```java
// NeighborHood.java:57-134
public static NeighborHood[] computeNeighborhoods(
    float[][] centroids,
    VectorSimilarityFunction similarityFunction
) {
    int k = centroids.length;

    if (k <= 10_000) {
        // Small k: brute force O(k²)
        return computeNeighborhoodsBruteForce(centroids, similarityFunction);
    } else {
        // Large k: HNSW graph-based O(k log k)
        return computeNeighborhoodsGraph(centroids, similarityFunction);
    }
}
```

### Usage in Clustering

Neighborhoods enable local search during assignment:

```java
// KMeansLocal.java:174-225
private static int[] assignWithNeighborhood(
    FloatVectorValues vectors,
    float[][] centroids,
    NeighborHood[] neighborhoods,
    int[] previousAssignments
) throws IOException {
    int[] newAssignments = new int[vectors.size()];

    for (int i = 0; i < vectors.size(); i++) {
        float[] vector = vectors.vectorValue(i);
        int currentCentroid = previousAssignments[i];

        // Only search within neighborhood of current centroid
        int[] candidates = neighborhoods[currentCentroid].neighbors();

        float bestScore = Float.NEGATIVE_INFINITY;
        int bestCentroid = currentCentroid;

        for (int candidateOrd : candidates) {
            float score = VectorUtil.dotProduct(vector, centroids[candidateOrd]);
            if (score > bestScore) {
                bestScore = score;
                bestCentroid = candidateOrd;
            }
        }

        newAssignments[i] = bestCentroid;
    }

    return newAssignments;
}
```

---

## SOAR (Spilled from Nearby Neighborhoods)

### Purpose

Vectors near cluster boundaries may have multiple good centroids. SOAR provides secondary assignments for better recall.

### SOAR Distance Formula

```java
// KMeansLocal.java:261-268
// soar(x, c) = ||x - c||² + lambda * ((x - c₁)ᵗ(x - c))² / ||x - c₁||²
//
// Where:
//   x = document vector
//   c = candidate centroid
//   c₁ = primary assigned centroid
//   lambda = SOAR weight parameter (default 1.0)
```

### Computing SOAR Assignments

```java
// KMeansLocal.java:253-341
private static int[] assignSpilledSlice(
    FloatVectorValues vectors,
    float[][] centroids,
    NeighborHood[] neighborhoods,
    int[] assignments,
    float lambda,
    int sliceStart,
    int sliceEnd
) throws IOException {
    int[] soarAssignments = new int[sliceEnd - sliceStart];
    Arrays.fill(soarAssignments, NO_SOAR_ASSIGNMENT);

    for (int i = sliceStart; i < sliceEnd; i++) {
        float[] vector = vectors.vectorValue(i);
        int primaryCentroid = assignments[i];
        float[] c1 = centroids[primaryCentroid];

        // Check if vector well-represented by primary
        float distanceToPrimary = VectorUtil.squareDistance(vector, c1);
        float threshold = neighborhoods[primaryCentroid].maxIntraDistance();

        if (distanceToPrimary <= threshold) {
            // Well represented - no SOAR needed
            continue;
        }

        // Find best SOAR assignment using modified distance
        int[] candidates = neighborhoods[primaryCentroid].neighbors();
        float bestSoarScore = Float.POSITIVE_INFINITY;
        int bestSoarCentroid = NO_SOAR_ASSIGNMENT;

        for (int candidateOrd : candidates) {
            if (candidateOrd == primaryCentroid) continue;

            float[] c = centroids[candidateOrd];
            float distToCandidate = VectorUtil.squareDistance(vector, c);

            // Calculate SOAR penalty term
            float xMinusC1DotXMinusC = 0;
            for (int d = 0; d < vector.length; d++) {
                float xMinusC1 = vector[d] - c1[d];
                float xMinusC = vector[d] - c[d];
                xMinusC1DotXMinusC += xMinusC1 * xMinusC;
            }
            float penalty = lambda * (xMinusC1DotXMinusC * xMinusC1DotXMinusC) / distanceToPrimary;

            float soarDist = distToCandidate + penalty;

            if (soarDist < bestSoarScore) {
                bestSoarScore = soarDist;
                bestSoarCentroid = candidateOrd;
            }
        }

        soarAssignments[i - sliceStart] = bestSoarCentroid;
    }

    return soarAssignments;
}
```

---

## Key Code References

### Clustering Algorithm
- **HierarchicalKMeans.cluster()**: `server/src/main/java/org/elasticsearch/index/codec/vectors/cluster/HierarchicalKMeans.java:114-144`
- **clusterAndSplit()**: Lines 146-226
- **updateAssignmentsWithRecursiveSplit()**: Lines 260-302
- **KMeansLocal.cluster()**: `server/src/main/java/org/elasticsearch/index/codec/vectors/cluster/KMeansLocal.java:370-393`

### Parent/Child Writing
- **buildCentroidGroups()**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsWriter.java:506-533`
- **writeCentroidsWithParents()**: Lines 435-478
- **writeCentroidsWithoutParents()**: Lines 480-502
- **Parent threshold decision**: Line 399

### Parent/Child Reading
- **getCentroidIterator()**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsReader.java:112-193`
- **getCentroidIteratorWithParents()**: Lines 310-441
- **getCentroidIteratorNoParent()**: Lines 266-308
- **Parent detection**: Line 163

### Supporting Data Structures
- **NeighborHood**: `server/src/main/java/org/elasticsearch/index/codec/vectors/cluster/NeighborHood.java:32`
- **KMeansIntermediate**: `server/src/main/java/org/elasticsearch/index/codec/vectors/cluster/KMeansIntermediate.java:17-41`
- **KMeansResult**: `server/src/main/java/org/elasticsearch/index/codec/vectors/cluster/KMeansResult.java:15-48`

---

## Summary

1. **Hierarchical k-means recursively splits** oversized clusters (>134% target) until all clusters are appropriately sized
2. **Parent references are implicit**, determined by centroid count threshold: `numCentroids > centroidsPerParentCluster²`
3. **Tree is flattened** into linear arrays with ordinal remapping during recursion
4. **Two-level hierarchy** is constructed on-the-fly during writes and reconstructed during reads
5. **Search traversal** scores parents first, then lazily scores children of best parents
6. **SOAR assignments** improve recall for boundary vectors by providing secondary cluster assignments
7. **Neighborhood awareness** enables efficient local search during refinement
