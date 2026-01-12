# Elasticsearch Vector Search: Complete Read Flow

## Overview

This document provides a comprehensive walkthrough of how data is read during vector search in Elasticsearch, from query submission to result collection.

## Table of Contents
1. [Complete Search Flow](#complete-search-flow)
2. [Centroid Reading and Scoring](#centroid-reading-and-scoring)
3. [Posting List Access and Processing](#posting-list-access-and-processing)
4. [Vector Reading and Scoring](#vector-reading-and-scoring)
5. [Order of Operations](#order-of-operations)
6. [Critical Data Structures](#critical-data-structures)
7. [Performance Optimizations](#performance-optimizations)
8. [Key Code References](#key-code-references)

---

## Complete Search Flow

### Phase 1: Query Rewrite & Initialization

**Entry Point:** `AbstractIVFKnnVectorQuery.rewrite()`
- **File:** `server/src/main/java/org/elasticsearch/search/vectors/AbstractIVFKnnVectorQuery.java`
- **Lines:** 100-160

```java
@Override
public Query rewrite(IndexSearcher indexSearcher) throws IOException {
    // Step 1: Get index reader contexts (segments)
    List<LeafReaderContext> leafReaderContexts = indexSearcher.getIndexReader().leaves();

    // Step 2: Calculate visit ratio (if not provided)
    float visitRatio = this.providedVisitRatio;
    if (visitRatio == 0.0f) {
        long totalVectors = calculateTotalVectors(leafReaderContexts);
        int numCands = Math.min(10_000, Math.max(numCandidates, 5 * k));
        float expected = (float) (Math.log10(totalVectors) * Math.log10(totalVectors) * numCands);
        visitRatio = expected / totalVectors;
    }

    // Step 3: Create filter weight (if filter query exists)
    Weight filterWeight = null;
    if (filter != null) {
        filterWeight = indexSearcher.createWeight(
            indexSearcher.rewrite(filter),
            ScoreMode.COMPLETE_NO_SCORES,
            1f
        );
    }

    // Step 4: Create collector manager (2*k to account for duplicates)
    IVFCollectorManager knnCollectorManager = getKnnCollectorManager(
        Math.round(2f * k),
        indexSearcher
    );

    // Step 5: Create parallel search tasks (one per segment)
    List<Callable<TopDocs>> tasks = new ArrayList<>(leafReaderContexts.size());
    for (LeafReaderContext context : leafReaderContexts) {
        tasks.add(() -> searchLeaf(context, filterWeight, knnCollectorManager, visitRatio));
    }

    // Step 6: Execute searches in parallel
    TopDocs[] perLeafResults = taskExecutor.invokeAll(tasks).toArray(TopDocs[]::new);

    // Step 7: Merge results from all segments
    TopDocs merged = TopDocs.merge(k, perLeafResults);

    // Step 8: Return constant score query with doc IDs
    return createRewrittenQuery(indexSearcher.getIndexReader(), merged);
}
```

### Phase 2: Per-Leaf Search Execution

**Method:** `IVFKnnFloatVectorQuery.approximateSearch()`
- **File:** `server/src/main/java/org/elasticsearch/search/vectors/IVFKnnFloatVectorQuery.java`
- **Lines:** 78-103

```java
@Override
public TopDocs approximateSearch(
    LeafReaderContext context,
    AcceptDocs acceptDocs,
    int visitedLimit
) throws IOException {
    // Get per-segment vector reader
    LeafReader reader = context.reader();
    KnnVectorReader vectorReader = reader.getVectorReader();

    if (!(vectorReader instanceof IVFVectorsReader ivfReader)) {
        // Fallback to exact search
        return exactSearch(context, acceptDocs);
    }

    // Create KNN collector for this segment
    KnnCollector knnCollector = knnCollectorManager.newCollector(visitedLimit, context);

    // Delegate to IVF reader
    ivfReader.searchNearestVectors(field, query, knnCollector, acceptDocs);

    // Return top results
    return knnCollector.topDocs();
}
```

### Phase 3: IVF Vector Reader Search

**Method:** `IVFVectorsReader.search()`
- **File:** `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/IVFVectorsReader.java`
- **Lines:** 268-392

```java
@Override
public final void search(
    String field,
    float[] target,
    KnnCollector knnCollector,
    AcceptDocs acceptDocs
) throws IOException {
    FieldInfo fieldInfo = fieldInfos.fieldInfo(field);

    // Step 1: Validate dimensions
    if (target.length != fieldInfo.getVectorDimension()) {
        throw new IllegalArgumentException(
            "Query vector dimension " + target.length +
            " does not match field dimension " + fieldInfo.getVectorDimension()
        );
    }

    // Step 2: Get field entry and calculate costs
    FieldEntry entry = fields.get(fieldInfo.number);
    int numVectors = entry.numVectors();

    // Calculate filtering cost
    long approximateCost = acceptDocs == null ? numVectors : acceptDocs.cost();
    float percentFiltered = (float) approximateCost / numVectors;

    // Step 3: Determine visit ratio
    float visitRatio = knnCollector.getSearchStrategy() != null
        ? knnCollector.getSearchStrategy().getVisitRatio()
        : calculateDynamicVisitRatio(numVectors, knnCollector.k());

    long maxVectorVisited = Math.round(numVectors * visitRatio);

    // Step 4: Get centroid iterator (scores all centroids)
    IndexInput centroidSlice = entry.centroidSlice(ivfCentroids);
    IndexInput postListSlice = entry.postingListSlice(ivfClusters);

    CentroidIterator centroidIterator = getCentroidIterator(
        fieldInfo,
        entry.numCentroids,
        centroidSlice,
        target,
        postListSlice,
        acceptDocs,
        approximateCost,
        values,
        visitRatio
    );

    // Step 5: Choose execution strategy
    boolean useParallelEvaluation = this instanceof ESNextDiskBBQVectorsReader;

    if (useParallelEvaluation) {
        // NEW: Parallel posting list evaluation
        searchWithParallelPostingLists(
            fieldInfo,
            postListSlice,
            target,
            acceptDocs,
            centroidIterator,
            knnCollector,
            maxVectorVisited,
            maxDoc,
            hasOverspillAssignments,
            numVectors,
            percentFiltered
        );
    } else {
        // FALLBACK: Sequential posting list evaluation
        searchSequential(
            fieldInfo,
            postListSlice,
            target,
            acceptDocs,
            centroidIterator,
            knnCollector,
            maxVectorVisited,
            numVectors,
            percentFiltered
        );
    }
}
```

---

## Centroid Reading and Scoring

### Step A: Centroid Iterator Creation

**Method:** `ESNextDiskBBQVectorsReader.getCentroidIterator()`
- **File:** `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsReader.java`
- **Lines:** 113-194

```java
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
    // Step 1: Build centroid filter bitset (optional optimization)
    FixedBitSet acceptCentroids = null;
    float approximateDocsPerCentroid = approximateCost / numCentroids;

    if (approximateDocsPerCentroid > 1.25 || numCentroids == 1) {
        // Skip filtering - most centroids will have matches
        acceptCentroids = null;
    } else {
        // Build bitset of centroids containing matching docs
        acceptCentroids = new FixedBitSet(numCentroids);

        // Intersect acceptDocs with vector doc IDs
        DocIdSetIterator acceptDocsIter = acceptDocs.iterator();
        DocIdSetIterator vectorDocsIter = getDocIndexIterator(values);
        DocIdSetIterator intersection = ConjunctionUtils.intersectIterators(
            List.of(acceptDocsIter, vectorDocsIter)
        );

        // Mark centroids that have matching documents
        int doc;
        while ((doc = intersection.nextDoc()) != NO_MORE_DOCS) {
            int centroidOrd = getCentroidOrdinalForDoc(doc);
            acceptCentroids.set(centroidOrd);
        }
    }

    // Step 2: Quantize query vector
    int dimension = fieldInfo.getVectorDimension();
    byte[] quantized = new byte[dimension];
    float[] globalCentroid = entry.globalCentroid;

    OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer();
    OptimizedScalarQuantizer.QuantizationResult queryParams = quantizer.scalarQuantize(
        targetQuery,
        new float[dimension],
        new int[dimension],
        (byte) 7,  // 7-bit quantization for centroids
        globalCentroid
    );

    // Convert quantized ints to bytes
    for (int i = 0; i < dimension; i++) {
        quantized[i] = (byte) queryParams.quantized()[i];
    }

    // Step 3: Create scorer for bulk centroid scoring
    ES92Int7VectorsScorer scorer = new ES92Int7VectorsScorer();

    // Step 4: Read parent count to determine structure
    int numParents = centroids.readVInt();

    CentroidIterator centroidIterator;
    if (numParents > 0) {
        // Hierarchical structure: score parents, then children
        centroidIterator = getCentroidIteratorWithParents(
            fieldInfo,
            centroids,
            numParents,
            numCentroids,
            scorer,
            quantized,
            queryParams,
            entry.globalCentroidDp,
            visitRatio * centroidOversampling,  // Oversample for recall
            acceptCentroids
        );
    } else {
        // Flat structure: score all centroids
        centroidIterator = getCentroidIteratorNoParent(
            fieldInfo,
            centroids,
            numCentroids,
            scorer,
            quantized,
            queryParams,
            entry.globalCentroidDp,
            acceptCentroids
        );
    }

    // Step 5: Wrap with prefetching iterator
    return getPostingListPrefetchIterator(centroidIterator, postingListSlice);
}
```

### Step B: Flat Centroid Scoring (No Parents)

**Method:** `getCentroidIteratorNoParent()`
- **Lines:** 266-308

```java
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

    // Calculate centroid size (quantized vector + corrections)
    int dimension = fieldInfo.getVectorDimension();
    final long centroidQuantizeSize = dimension + 3 * Float.BYTES + Integer.BYTES;

    // Score all centroids in bulk (16 at a time)
    score(
        neighborQueue,
        numCentroids,
        0,  // offset
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

    // Return iterator that yields centroids in score order
    return new CentroidIterator() {
        @Override
        public boolean hasNext() {
            return neighborQueue.size() > 0;
        }

        @Override
        public CentroidMeta nextCentroidMeta() throws IOException {
            int centroidOrd = neighborQueue.pop();

            // Seek to posting list offset for this centroid
            long offsetsPosition = centroids.getFilePointer() +
                (long) (centroidOrd - alreadyScored) * centroidQuantizeSize +
                (long) numCentroids * centroidQuantizeSize;

            centroids.seek(offsetsPosition);
            long postingListOffset = centroids.readLong();

            return new CentroidMeta(centroidOrd, postingListOffset);
        }
    };
}
```

### Bulk Centroid Scoring

**Method:** `score()`
- **Lines:** 477-542

```java
private static void score(
    NeighborQueue neighborQueue,
    int size,
    int scoresOffset,
    ES92Int7VectorsScorer scorer,
    IndexInput centroids,
    long centroidQuantizeSize,
    byte[] quantizeQuery,
    OptimizedScalarQuantizer.QuantizationResult queryCorrections,
    float centroidDp,
    VectorSimilarityFunction similarityFunction,
    float[] scores,
    FixedBitSet acceptCentroids
) throws IOException {
    final int BULK_SIZE = ES92Int7VectorsScorer.BULK_SIZE;  // 16
    int limit = size - BULK_SIZE + 1;

    // Process full bulks (16 centroids at a time)
    for (int i = 0; i < limit; i += BULK_SIZE) {
        // Check if any centroid in this bulk passes filter
        if (acceptCentroids != null &&
            acceptCentroids.cardinality(scoresOffset + i, scoresOffset + i + BULK_SIZE) == 0) {
            // Skip entire bulk - no matching centroids
            centroids.skipBytes(BULK_SIZE * centroidQuantizeSize);
            continue;
        }

        // Bulk score 16 centroids using SIMD
        scorer.scoreBulk(
            quantizeQuery,
            queryCorrections.lowerInterval(),
            queryCorrections.upperInterval(),
            queryCorrections.quantizedComponentSum(),
            queryCorrections.additionalCorrection(),
            similarityFunction,
            centroidDp,
            scores,
            BULK_SIZE
        );

        // Add scored centroids to priority queue
        for (int j = 0; j < BULK_SIZE; j++) {
            int centroidOrd = scoresOffset + i + j;

            if (acceptCentroids == null || acceptCentroids.get(centroidOrd)) {
                neighborQueue.add(centroidOrd, scores[j]);
            }
        }
    }

    // Handle tail (remaining < BULK_SIZE centroids)
    if (limit < size) {
        int remaining = size - limit;
        scorer.scoreBulk(
            quantizeQuery,
            queryCorrections.lowerInterval(),
            queryCorrections.upperInterval(),
            queryCorrections.quantizedComponentSum(),
            queryCorrections.additionalCorrection(),
            similarityFunction,
            centroidDp,
            scores,
            remaining
        );

        for (int j = 0; j < remaining; j++) {
            int centroidOrd = scoresOffset + limit + j;
            if (acceptCentroids == null || acceptCentroids.get(centroidOrd)) {
                neighborQueue.add(centroidOrd, scores[j]);
            }
        }
    }
}
```

### Step C: Hierarchical Centroid Scoring (With Parents)

**Method:** `getCentroidIteratorWithParents()`
- **Lines:** 310-441

This implements a two-stage scoring strategy:

1. **Score all parent centroids** (e.g., 256 parents)
2. **Lazily score children** of best parents as needed

```java
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

    // Three queues for managing hierarchical traversal
    final NeighborQueue parentsQueue = new NeighborQueue(numToVisit, true);
    final NeighborQueue currentParentQueue = new NeighborQueue(numToVisit, true);
    final NeighborQueue neighborQueue = new NeighborQueue(numToVisit, true);

    // Read parent metadata header
    int maxChildrenPerParent = centroids.readVInt();
    long parentMetadataOffset = centroids.getFilePointer() +
        (long) numParents * (dimension + 3 * Float.BYTES + Integer.BYTES);

    // Score all parent centroids (bulk SIMD)
    long centroidQuantizeSize = dimension + 3 * Float.BYTES + Integer.BYTES;
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

    // Calculate children metadata offset
    long childrenOffset = parentMetadataOffset + (2L * Integer.BYTES * numParents);

    // Populate initial children from best parent
    if (parentsQueue.size() > 0) {
        int bestParent = parentsQueue.pop();
        populateChildrenFromParent(
            centroids,
            parentMetadataOffset,
            childrenOffset,
            bestParent,
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

    // Return iterator that lazily loads more children
    return new CentroidIterator() {
        @Override
        public boolean hasNext() {
            return neighborQueue.size() > 0 ||
                   currentParentQueue.size() > 0 ||
                   parentsQueue.size() > 0;
        }

        @Override
        public CentroidMeta nextCentroidMeta() throws IOException {
            // Load next parent's children if current exhausted
            if (neighborQueue.size() == 0 &&
                currentParentQueue.size() == 0 &&
                parentsQueue.size() > 0) {

                int nextParent = parentsQueue.pop();
                populateChildrenFromParent(
                    centroids,
                    parentMetadataOffset,
                    childrenOffset,
                    nextParent,
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

            // Move child from currentParent to neighbor queue
            if (neighborQueue.size() == 0 && currentParentQueue.size() > 0) {
                int childOrd = currentParentQueue.pop();
                neighborQueue.add(childOrd, 0);  // Already scored
            }

            // Return next child centroid
            int centroidOrd = neighborQueue.pop();
            long offset = readChildPostingListOffset(
                centroids,
                childrenOffset,
                centroidOrd
            );

            return new CentroidMeta(centroidOrd, offset);
        }
    };
}
```

### Step D: Prefetching Iterator

**Method:** `getPostingListPrefetchIterator()`
- **Lines:** 59-93

Wraps the centroid iterator to prefetch posting list data for the next centroid:

```java
private static CentroidIterator getPostingListPrefetchIterator(
    CentroidIterator centroidIterator,
    IndexInput postingListSlice
) throws IOException {
    return new CentroidIterator() {
        private CentroidMeta next = null;

        @Override
        public boolean hasNext() {
            if (next == null && centroidIterator.hasNext()) {
                try {
                    next = centroidIterator.nextCentroidMeta();
                    prefetch(next);  // Prefetch data for this centroid
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
            return next != null;
        }

        @Override
        public CentroidMeta nextCentroidMeta() throws IOException {
            CentroidMeta result = next;
            next = null;
            return result;
        }

        private void prefetch(CentroidMeta centroid) throws IOException {
            // Advise OS to prefetch posting list data
            postingListSlice.prefetch(centroid.offset(), PREFETCH_SIZE);
        }
    };
}
```

---

## Posting List Access and Processing

### Sequential Processing (Fallback)

**Method:** `IVFVectorsReader.search()` - Sequential block
- **Lines:** 357-391

```java
// Phase 1: Initial exploration
PostingVisitor scorer = getPostingVisitor(fieldInfo, postListSlice, target, acceptDocs);
long expectedDocs = 0;
long actualDocs = 0;

while (centroidIterator.hasNext() &&
       (maxVectorVisited > expectedDocs ||
        knnCollector.minCompetitiveSimilarity() == Float.NEGATIVE_INFINITY)) {

    // Get next best centroid
    CentroidMeta centroidMeta = centroidIterator.nextCentroidMeta();

    // Reset scorer to this posting list
    expectedDocs += scorer.resetPostingsScorer(centroidMeta.offset());

    // Visit and score all vectors in this posting list
    actualDocs += scorer.visit(knnCollector);

    // Update search strategy
    if (knnCollector.getSearchStrategy() != null) {
        knnCollector.getSearchStrategy().nextVectorsBlock();
    }

    // Early termination check
    if (knnCollector.earlyTerminated()) {
        return;
    }
}

// Phase 2: Filter compensation (if filtering active)
if (acceptDocs != null) {
    float unfilteredRatioVisited = (float) expectedDocs / numVectors;
    int filteredVectors = (int) Math.ceil(numVectors * percentFiltered);
    float expectedScored = Math.min(
        2 * filteredVectors * unfilteredRatioVisited,
        expectedDocs / 2f
    );

    while (centroidIterator.hasNext() &&
           (actualDocs < expectedScored || actualDocs < knnCollector.k())) {
        CentroidMeta centroidMeta = centroidIterator.nextCentroidMeta();
        expectedDocs += scorer.resetPostingsScorer(centroidMeta.offset());
        actualDocs += scorer.visit(knnCollector);

        if (knnCollector.earlyTerminated()) {
            return;
        }
    }
}
```

### Parallel Processing (New Approach)

**Method:** `IVFVectorsReader.searchWithParallelPostingLists()`
- **Lines:** 409-551

Processes multiple posting lists in batches for better efficiency:

```java
private void searchWithParallelPostingLists(
    FieldInfo fieldInfo,
    IndexInput postingListInput,
    float[] target,
    AcceptDocs acceptDocs,
    CentroidIterator centroidIterator,
    KnnCollector knnCollector,
    long maxVectorVisited,
    int maxDoc,
    boolean hasOverspillAssignments,
    int numVectors,
    float percentFiltered
) throws IOException {
    final int BATCH_SIZE = 8;  // Process 8 posting lists at once

    // Initialize deduplication filter for overspill handling
    final IncrementalDeduplicationFilter deduplicationFilter =
        new IncrementalDeduplicationFilter(maxDoc);
    if (hasOverspillAssignments) {
        deduplicationFilter.initialize();
    }

    // Initialize incremental filter iterator
    DocIdSetIterator filterIterator = acceptDocs == null ? null : acceptDocs.iterator();
    final IncrementalFilterIterator incrementalFilterIterator =
        new IncrementalFilterIterator(filterIterator, maxDoc);

    long expectedDocs = 0;
    long actualDocs = 0;

    // Phase 1: Initial exploration
    while (centroidIterator.hasNext() &&
           false == knnCollector.earlyTerminated() &&
           (expectedDocs < maxVectorVisited ||
            knnCollector.minCompetitiveSimilarity() == Float.NEGATIVE_INFINITY)) {

        // Step 1: Load batch of posting lists
        int batchCount = 0;
        PostingVisitor[] visitors = new PostingVisitor[BATCH_SIZE];
        int[] minDocIds = new int[BATCH_SIZE];

        while (centroidIterator.hasNext() && batchCount < BATCH_SIZE) {
            CentroidMeta centroid = centroidIterator.nextCentroidMeta();

            PostingVisitor visitor = getPostingVisitor(
                fieldInfo,
                postingListInput,
                target,
                null  // Don't pass acceptDocs - filtering done later
            );

            long size = visitor.resetPostingsScorer(centroid.offset());

            if (size > 0) {
                // Peek at first doc ID (without advancing)
                int minDocId = visitor.peekFirstDocId();

                visitors[batchCount] = visitor;
                minDocIds[batchCount] = minDocId;
                expectedDocs += size;
                batchCount++;
            }
        }

        if (batchCount == 0) {
            break;
        }

        // Step 2: Sort posting lists by min doc ID
        // (maintains doc ID order for incremental filter iterator)
        Integer[] indices = new Integer[batchCount];
        for (int i = 0; i < batchCount; i++) {
            indices[i] = i;
        }
        java.util.Arrays.sort(
            indices,
            0,
            batchCount,
            Comparator.comparingInt(a -> minDocIds[a])
        );

        // Step 3: Process posting lists in min doc ID order
        for (int idx = 0; idx < batchCount; idx++) {
            int i = indices[idx];
            PostingVisitor visitor = visitors[i];

            // Visit and score with post-filtering
            int scored = visitor.visitFiltered(
                knnCollector,
                incrementalFilterIterator,
                deduplicationFilter
            );

            actualDocs += scored;

            if (knnCollector.earlyTerminated()) {
                return;
            }
        }
    }

    // Phase 2: Filter compensation
    if (acceptDocs != null) {
        float unfilteredRatioVisited = (float) expectedDocs / numVectors;
        int filteredVectors = (int) Math.ceil(numVectors * percentFiltered);
        float expectedScored = Math.min(
            2 * filteredVectors * unfilteredRatioVisited,
            expectedDocs / 2f
        );

        while (centroidIterator.hasNext() &&
               false == knnCollector.earlyTerminated() &&
               (actualDocs < expectedScored || actualDocs < knnCollector.k())) {

            // Load next batch and process...
            // (similar to Phase 1)
        }
    }
}
```

---

## Vector Reading and Scoring

### Posting Visitor Structure

**Class:** `MemorySegmentPostingsVisitor`
- **File:** `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsReader.java`
- **Lines:** 559-1222

### Initializing Posting List Reader

**Method:** `resetPostingsScorer()`
- **Lines:** 629-688

```java
@Override
public long resetPostingsScorer(long offset) throws IOException {
    // Seek to posting list start
    indexInput.seek(offset);
    slicePos = offset;

    // Read raw centroid (full precision float32)
    int dimension = fieldInfo.getVectorDimension();
    for (int i = 0; i < dimension; i++) {
        centroid[i] = Float.intBitsToFloat(indexInput.readInt());
    }

    // Read centroid dot product
    centroidDp = Float.intBitsToFloat(indexInput.readInt());

    // Read vector count
    vectors = indexInput.readVInt();

    if (vectors == 0) {
        return 0;
    }

    // Read doc ID encoding strategy
    docEncoding = indexInput.readByte();

    // Initialize doc ID reader
    idsWriter = createDocIdsWriter(docEncoding);

    // Calculate quantized vector size
    quantizedVectorByteSize = quantEncoding.getDocPackedLength(dimension);

    return vectors;
}
```

### Sequential Vector Scoring

**Method:** `visit()`
- **Lines:** 721-793

```java
@Override
public int visit(KnnCollector knnCollector) throws IOException {
    indexInput.seek(slicePos);
    docBase = 0;
    int scoredDocs = 0;

    // Quantize query if needed
    quantizeQueryIfNecessary();

    // Process full batches (BULK_SIZE = 16)
    int limit = vectors - BULK_SIZE + 1;
    int i = 0;

    for (; i < limit; i += BULK_SIZE) {
        // Read doc IDs for batch
        readDocIds(BULK_SIZE);

        // Check how many pass filter
        final int docsToBulkScore = acceptDocs == null
            ? BULK_SIZE
            : docToBulkScore(docIdsScratch, acceptDocs);

        if (docsToBulkScore == 0) {
            // Skip scoring - all filtered out
            indexInput.skipBytes(quantizedByteLength * BULK_SIZE);
            continue;
        }

        // Choose scoring strategy
        final float maxScore;
        if (docsToBulkScore < BULK_SIZE / 2) {
            // Most docs filtered - score individually
            maxScore = scoreIndividually();
        } else {
            // Bulk score with SIMD
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

        // Collect results above threshold
        if (knnCollector.minCompetitiveSimilarity() < maxScore) {
            collectBulk(knnCollector, scores);
        }

        scoredDocs += docsToBulkScore;
        knnCollector.incVisitedCount(BULK_SIZE);
    }

    // Handle tail (< BULK_SIZE vectors)
    if (i < vectors) {
        int remaining = vectors - i;
        readDocIds(remaining);

        for (int j = 0; j < remaining; j++) {
            if (acceptDocs == null || acceptDocs.bits().get(docIdsScratch[j])) {
                scoreAndCollectIndividual(j, knnCollector);
                scoredDocs++;
            }
        }

        knnCollector.incVisitedCount(remaining);
    }

    return scoredDocs;
}
```

### Bulk Vector Scoring with Post-Filtering

**Method:** `scoreBulkWithPostFiltering()`
- **Lines:** 977-1079

This is the new efficient approach used in parallel posting list evaluation:

```java
@Override
public int scoreBulkWithPostFiltering(
    int[] docIds,
    int count,
    IncrementalFilterIterator filterIterator,
    IncrementalDeduplicationFilter deduplicationFilter,
    KnnCollector knnCollector
) throws IOException {
    if (count == 0 || currentBatchSize == 0) {
        return 0;
    }

    // Quantize query if needed
    quantizeQueryIfNecessary();

    int collected = 0;

    if (currentBatchSize == BULK_SIZE) {
        // Full batch: use bulk SIMD scoring

        // Score ALL docs at once (no pre-filtering)
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

        // POST-FILTER: Apply filters to scored results
        if (knnCollector.minCompetitiveSimilarity() < maxScore) {
            for (int i = 0; i < currentBatchSize; i++) {
                int docId = docIdsScratch[i];

                // Check filter
                if (filterIterator.matches(docId) == false) {
                    continue;
                }

                // Check deduplication
                if (deduplicationFilter.alreadySeen(docId)) {
                    continue;
                }

                // Mark as seen and collect
                deduplicationFilter.markSeen(docId);
                knnCollector.collect(docId, scores[i]);
                collected++;
            }
        }

        knnCollector.incVisitedCount(currentBatchSize);

    } else {
        // Partial batch: score individually but still apply post-filtering
        for (int i = 0; i < currentBatchSize; i++) {
            int docId = docIdsScratch[i];

            // Check filter
            if (filterIterator.matches(docId) == false) {
                indexInput.skipBytes(quantizedVectorByteSize);
                continue;
            }

            // Check deduplication
            if (deduplicationFilter.alreadySeen(docId)) {
                indexInput.skipBytes(quantizedVectorByteSize);
                continue;
            }

            // Score individual vector
            float score = scoreIndividual();

            // Collect if above threshold
            if (score >= knnCollector.minCompetitiveSimilarity()) {
                deduplicationFilter.markSeen(docId);
                knnCollector.collect(docId, score);
                collected++;
            }
        }

        knnCollector.incVisitedCount(currentBatchSize);
    }

    return collected;
}
```

### Individual Vector Scoring

**Method:** `scoreIndividually()`
- **Lines:** 646-688

```java
private float scoreIndividually() throws IOException {
    float maxScore = Float.NEGATIVE_INFINITY;

    // Read and score each quantized vector
    for (int j = 0; j < BULK_SIZE; j++) {
        int doc = docIdsScratch[j];

        if (doc != -1) {
            // Read quantized vector
            indexInput.readBytes(quantizedVectorScratch, 0, quantizedVectorByteSize);

            // Compute quantized score (bit-level operations)
            float qcDist = osqVectorsScorer.quantizeScore(quantizedVectorScratch);
            scores[j] = qcDist;
        } else {
            // Skip this position
            indexInput.skipBytes(quantizedVectorByteSize);
        }
    }

    // Read all corrections in bulk
    indexInput.readFloats(correctionsLower, 0, BULK_SIZE);
    indexInput.readFloats(correctionsUpper, 0, BULK_SIZE);
    for (int j = 0; j < BULK_SIZE; j++) {
        correctionsSum[j] = Short.toUnsignedInt(indexInput.readShort());
    }
    indexInput.readFloats(correctionsAdd, 0, BULK_SIZE);

    // Apply corrections to get final scores
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
                scores[j]
            );

            if (scores[j] > maxScore) {
                maxScore = scores[j];
            }
        }
    }

    return maxScore;
}
```

### Reading Doc IDs

**Method:** `readDocIds()`
- **Lines:** 795-815

```java
private void readDocIds(int count) throws IOException {
    // Read doc IDs from encoded format
    idsWriter.readInts(indexInput, count, docEncoding, docIdsScratch);

    // Convert deltas to absolute doc IDs
    for (int j = 0; j < count; j++) {
        docBase += docIdsScratch[j];  // Accumulate delta
        docIdsScratch[j] = docBase;   // Store absolute doc ID
    }
}
```

---

## Order of Operations

### Complete Execution Sequence

```
1. USER QUERY SUBMISSION
   └─> AbstractIVFKnnVectorQuery.rewrite()
       ├─ Calculate visitRatio
       ├─ Create IVFCollectorManager (2*k capacity)
       └─ Launch parallel tasks (one per segment)

           FOR EACH SEGMENT (parallel):
           └─> IVFKnnFloatVectorQuery.approximateSearch()
               └─> IVFVectorsReader.search()
                   │
                   ├─ STEP 1: Validate & Setup
                   │   ├─ Check vector dimensions
                   │   ├─ Calculate approximateCost (filtering)
                   │   └─ Determine visitRatio
                   │
                   ├─ STEP 2: Get Centroid Iterator
                   │   └─> ESNextDiskBBQVectorsReader.getCentroidIterator()
                   │       ├─ Build acceptCentroids bitset (filter centroids)
                   │       ├─ Quantize query vector (7-bit for centroids)
                   │       ├─ Read parent count
                   │       │
                   │       ├─ IF hierarchical:
                   │       │   └─> getCentroidIteratorWithParents()
                   │       │       ├─ Score ALL parent centroids (bulk SIMD)
                   │       │       ├─ Add to parentsQueue (priority queue)
                   │       │       ├─ Pop best parent
                   │       │       ├─ Score its children (bulk SIMD)
                   │       │       └─ Lazily load more parents as needed
                   │       │
                   │       └─ ELSE flat:
                   │           └─> getCentroidIteratorNoParent()
                   │               ├─ Score ALL centroids (bulk SIMD)
                   │               └─ Return in score order
                   │
                   └─ STEP 3: Choose Execution Strategy
                       │
                       ├─ IF ESNextDiskBBQVectorsReader:
                       │   └─> searchWithParallelPostingLists()
                       │       ├─ PHASE 1: Initial exploration
                       │       │   ├─ LOOP: while not early terminated & expectedDocs < maxVectorVisited
                       │       │   │   ├─ Load batch of 8 posting lists
                       │       │   │   │   ├─ getPostingVisitor() per centroid
                       │       │   │   │   ├─ resetPostingsScorer(offset)
                       │       │   │   │   │   ├─ Read raw centroid
                       │       │   │   │   │   ├─ Read centroid DP
                       │       │   │   │   │   ├─ Read vector count
                       │       │   │   │   │   └─ Read doc encoding
                       │       │   │   │   └─ peekFirstDocId() (for sorting)
                       │       │   │   ├─ Sort posting lists by minDocId
                       │       │   │   └─ Process in doc ID order
                       │       │   │       └─> visitor.visitFiltered()
                       │       │   │           ├─ LOOP: for each 16-vector batch
                       │       │   │           │   ├─ readDocIds(16)
                       │       │   │           │   ├─ scoreBulk() [SIMD - all 16]
                       │       │   │           │   ├─ POST-FILTER each result:
                       │       │   │           │   │   ├─ filterIterator.matches()
                       │       │   │           │   │   ├─ dedup.alreadySeen()
                       │       │   │           │   │   └─ collect if passes
                       │       │   │           │   └─ incVisitedCount(16)
                       │       │   │           └─ Handle tail (< 16 vectors)
                       │       │   └─ Check early termination
                       │       │
                       │       └─ PHASE 2: Filter compensation
                       │           └─ Continue Phase 1 until actualDocs meets filtered expectations
                       │
                       └─ FALLBACK: Sequential processing
                           ├─ PHASE 1: Initial exploration
                           │   └─ LOOP: while hasNext & expectedDocs < maxVectorVisited
                           │       ├─ getPostingVisitor()
                           │       ├─ resetPostingsScorer()
                           │       └─> visitor.visit()
                           │           ├─ LOOP: for each 16-vector batch
                           │           │   ├─ readDocIds(16)
                           │           │   ├─ Pre-filter: count docs passing acceptDocs
                           │           │   ├─ IF < 50% pass: scoreIndividually()
                           │           │   │   ELSE: scoreBulk() [SIMD]
                           │           │   └─ collectBulk()
                           │           └─ Handle tail
                           │
                           └─ PHASE 2: Filter compensation

2. RESULT COLLECTION & DEDUPLICATION
   └─> AbstractIVFKnnVectorQuery.searchLeaf()
       ├─ Get TopDocs from collector
       ├─ Deduplicate at per-leaf level
       └─ Return deduped results

3. MERGE RESULTS ACROSS SEGMENTS
   └─> TopDocs.merge(k, perLeafResults)
       └─> Return final k nearest neighbors

4. RETURN CONSTANT SCORE QUERY
   └─> createRewrittenQuery(reader, merged)
```

---

## Critical Data Structures

### IncrementalDeduplicationFilter

**File:** `server/src/main/java/org/elasticsearch/search/vectors/IncrementalDeduplicationFilter.java`

**Purpose:** Track already-seen documents to handle overspill duplicates

```java
public final class IncrementalDeduplicationFilter {
    private SparseFixedBitSet seenDocs;
    private int seenCount;
    private final int maxDoc;

    public boolean alreadySeen(int docId) {
        return seenDocs != null && seenDocs.get(docId);
    }

    public void markSeen(int docId) {
        if (seenDocs == null) {
            initialize();
        }
        if (seenDocs.getAndSet(docId) == false) {
            seenCount++;
        }
    }

    public void initialize() {
        if (seenDocs == null) {
            seenDocs = new SparseFixedBitSet(maxDoc);
        }
    }
}
```

### IncrementalFilterIterator

**File:** `server/src/main/java/org/elasticsearch/search/vectors/IncrementalFilterIterator.java`

**Purpose:** Efficiently check filter matches while building backing bitset

```java
public final class IncrementalFilterIterator {
    private final DocIdSetIterator iterator;
    private final SparseFixedBitSet backingBitset;

    public boolean matches(int docId) throws IOException {
        if (iterator == null) {
            return true;
        }

        int currentPos = iterator.docID();

        // Already scanned past docId - check bitset
        if (docId < currentPos) {
            return backingBitset != null && backingBitset.get(docId);
        }

        // Advance iterator to docId, marking matches
        int nextDocId;
        while ((nextDocId = iterator.nextDoc()) < docId) {
            backingBitset.set(nextDocId);
        }

        if (nextDocId != DocIdSetIterator.NO_MORE_DOCS) {
            backingBitset.set(nextDocId);
        }

        return nextDocId == docId;
    }
}
```

**Critical Assumption:** Expects doc IDs to be queried in strictly increasing order!

### PostingListCursor (Future Enhancement)

**File:** `server/src/main/java/org/elasticsearch/search/vectors/PostingListCursor.java`

**Purpose:** Maintain state for processing posting lists in batches

```java
final class PostingListCursor implements Comparable<PostingListCursor> {
    private static final int BULK_SIZE = 16;

    private final IVFVectorsReader.PostingVisitor postingVisitor;
    private final IncrementalFilterIterator filterIterator;
    private final int[] docIdBatch;
    private int batchSize;
    private int currentDocId;
    private final int loadOrder;  // For prioritization

    @Override
    public int compareTo(PostingListCursor other) {
        // Sort by load order (proximity to query)
        int cmp = Integer.compare(this.loadOrder, other.loadOrder);
        if (cmp != 0) {
            return cmp;
        }
        // Tie-breaker: doc ID
        return Integer.compare(this.currentDocId, other.currentDocId);
    }

    int scoreBatch(
        KnnCollector collector,
        IncrementalDeduplicationFilter deduplicationFilter
    ) throws IOException {
        return postingVisitor.scoreBulkWithPostFiltering(
            docIdBatch,
            batchSize,
            filterIterator,
            deduplicationFilter,
            collector
        );
    }
}
```

---

## Performance Optimizations

### 1. Bulk Scoring (SIMD)

Score 16 vectors at once using SIMD instructions:

```java
// 16× faster than individual scoring
osqVectorsScorer.scoreBulk(query, ..., scores)  // SIMD
```

### 2. Quantization

7-bit quantized centroids for fast scoring with corrections:

```java
// 4.5× smaller, 10× faster scoring
byte[] quantized = quantize(centroid, 7);
```

### 3. Post-Filtering

Score all docs first, then filter (maintains bulk efficiency):

```java
// Score all 16
scoreBulk(...);

// Filter afterwards
for (int i = 0; i < 16; i++) {
    if (filter.matches(docIds[i]) && !dedup.alreadySeen(docIds[i])) {
        collect(docIds[i], scores[i]);
    }
}
```

### 4. Prefetching

Prefetch next posting list while processing current:

```java
prefetch(nextCentroid.offset(), PREFETCH_SIZE);
```

### 5. Early Termination

Stop when k results found and threshold reached:

```java
if (knnCollector.earlyTerminated()) {
    return;
}
```

### 6. Centroid Filtering

Skip centroids with no matching documents:

```java
if (acceptCentroids != null && acceptCentroids.get(centroidOrd) == false) {
    continue;  // Skip this centroid
}
```

### 7. Hierarchical Clustering

For large centroid counts, score parents then children lazily:

```java
// Score 256 parents, then only children of best 64
// Reduces centroid scoring from O(N) to O(√N + k·√N)
```

### 8. Parallel Leaf Search

Process different index segments in parallel:

```java
// One task per segment
List<Callable<TopDocs>> tasks = createTasksPerSegment();
TopDocs[] results = executor.invokeAll(tasks);
```

---

## Key Code References

### Query Entry
- **AbstractIVFKnnVectorQuery.rewrite()**: `server/src/main/java/org/elasticsearch/search/vectors/AbstractIVFKnnVectorQuery.java:100-160`
- **IVFKnnFloatVectorQuery.approximateSearch()**: `server/src/main/java/org/elasticsearch/search/vectors/IVFKnnFloatVectorQuery.java:78-103`

### Search Orchestration
- **IVFVectorsReader.search()**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/IVFVectorsReader.java:268-392`
- **searchWithParallelPostingLists()**: Lines 409-551

### Centroid Scoring
- **getCentroidIterator()**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsReader.java:113-194`
- **getCentroidIteratorWithParents()**: Lines 310-441
- **getCentroidIteratorNoParent()**: Lines 266-308
- **score()** (bulk centroid scoring): Lines 477-542

### Vector Scoring
- **resetPostingsScorer()**: `server/src/main/java/org/elasticsearch/index/codec/vectors/diskbbq/next/ESNextDiskBBQVectorsReader.java:629-688`
- **visit()** (sequential): Lines 721-793
- **scoreBulkWithPostFiltering()**: Lines 977-1079
- **scoreIndividually()**: Lines 646-688

### Supporting Classes
- **IncrementalDeduplicationFilter**: `server/src/main/java/org/elasticsearch/search/vectors/IncrementalDeduplicationFilter.java`
- **IncrementalFilterIterator**: `server/src/main/java/org/elasticsearch/search/vectors/IncrementalFilterIterator.java`
- **PostingListCursor**: `server/src/main/java/org/elasticsearch/search/vectors/PostingListCursor.java`

---

## Summary

1. **Three-phase search**: Query rewrite → Per-segment search → Result merging
2. **Centroid-first**: Score centroids, then process posting lists in order of relevance
3. **Bulk operations**: Score 16 vectors at once using SIMD for maximum throughput
4. **Post-filtering**: Score first, filter afterwards (maintains bulk efficiency)
5. **Incremental structures**: Lazily build filter bitsets as documents are accessed
6. **Early termination**: Stop as soon as k results with sufficient threshold are found
7. **Hierarchical optimization**: Two-level centroid structure reduces scoring cost
8. **Parallel processing**: Multiple posting lists and segments processed concurrently
