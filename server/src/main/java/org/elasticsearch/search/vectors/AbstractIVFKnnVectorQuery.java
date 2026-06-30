/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.elasticsearch.common.lucene.Lucene;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.index.codec.vectors.cluster.BulkNeighborQueue;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidIterator;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.PostingMetadata;
import org.elasticsearch.search.internal.ContextIndexSearcher;
import org.elasticsearch.search.profile.query.QueryProfiler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAccumulator;

import static org.elasticsearch.search.vectors.AbstractMaxScoreKnnCollector.LEAST_COMPETITIVE;

/**
 * Base query for IVF (DiskBBQ) approximate kNN search. Parallelism is pushed down to posting-list
 * granularity: rather than scoring one leaf per task, every selected posting list from every leaf is
 * flattened into a single closest-first ordering and scored by a fixed pool of workers, all sharing
 * one lock-free competitive frontier. This keeps the frontier maximally tight (so farther postings
 * prune hard) and balances load across segments of wildly different sizes.
 * <p>
 * The pipeline runs in three phases:
 * <ol>
 *   <li><b>Phase A</b> (parallel, per leaf): select the closest postings up to a per-leaf budget and
 *       capture their {@link PostingMetadata}. No vector scoring happens here, so the selection is a
 *       scoring-free approximation of {@link IVFVectorsReader#search}'s budget.</li>
 *   <li><b>Phase B</b> (parallel, fixed worker pool): workers pull postings from a single lock-free
 *       cursor over the closest-first ordering and score them through their own thread-confined
 *       per-leaf state.</li>
 *   <li><b>Phase C</b>: union per-(worker, leaf) results, dedup by global doc id keeping the highest
 *       score, then merge across leaves down to {@code k}.</li>
 * </ol>
 */
abstract class AbstractIVFKnnVectorQuery extends Query implements QueryProfilerProvider {

    static final TopDocs NO_RESULTS = TopDocsCollector.EMPTY_TOPDOCS;

    /**
     * Optional explicit override for the number of posting-list scoring workers, parsed from the
     * {@code es.knn.ivf.workers} system property. When {@code >= 1} it pins the worker count (so benchmarks can
     * align it with the configured search thread pool on a force-merged single segment); otherwise it is
     * {@code -1} and {@link #scoringParallelism(IndexSearcher)} sizes the pool from the searcher instead.
     */
    private static final int WORKER_OVERRIDE = resolveWorkerOverride();

    /**
     * How many global-queue chunks to target per worker. Keeping several chunks per worker means the shared cursor
     * still load-balances finely (faster workers steal more chunks) while each chunk is large enough to prefetch
     * ahead of scoring.
     */
    private static final int QUEUE_CHUNK_OVERSUBSCRIPTION = 4;

    /**
     * How many postings ahead each worker keeps prefetched while scoring. This bounds the number of outstanding
     * posting-list prefetches to a small window so a very large run (e.g. a force-merged single segment with many
     * selected postings) does not issue the whole run's I/O at once - which would let the earliest pages be evicted
     * before they are scored - while still keeping enough in flight to hide storage latency (relevant for
     * higher-latency backing stores such as the stateless shared blob cache) behind CPU scoring.
     */
    private static final int PREFETCH_DISTANCE = 8;

    protected final String field;
    protected final float providedVisitRatio;
    protected final int k;
    protected final int numCands;
    protected final Query filter;
    protected int vectorOpsCount;
    protected boolean doPrecondition;

    protected AbstractIVFKnnVectorQuery(String field, float visitRatio, int k, int numCands, Query filter, boolean doPrecondition) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be at least 1, got: " + k);
        }
        if (visitRatio < 0.0f || visitRatio > 1.0f) {
            throw new IllegalArgumentException("visitRatio must be between 0.0 and 1.0 (both inclusive), got: " + visitRatio);
        }
        if (numCands < k) {
            throw new IllegalArgumentException("numCands must be at least k, got: " + numCands);
        }
        this.field = field;
        this.providedVisitRatio = visitRatio;
        this.k = k;
        this.filter = filter;
        this.numCands = numCands;
        this.doPrecondition = doPrecondition;
    }

    private static int resolveWorkerOverride() {
        String prop = System.getProperty("es.knn.ivf.workers");
        if (prop != null && prop.isBlank() == false) {
            try {
                int parsed = Integer.parseInt(prop.trim());
                if (parsed >= 1) {
                    return parsed;
                }
            } catch (NumberFormatException ignored) {
                // fall through to the pool-aware default
            }
        }
        return -1;
    }

    /**
     * The number of posting-list scoring workers to run for this query. An explicit {@code es.knn.ivf.workers}
     * override wins when set. Otherwise, under a {@link ContextIndexSearcher} (the production search path) this is the
     * searcher's resolved maximum slice count, which is already sized to the search thread pool and falls back to
     * {@code 1} when the pool is saturated — so a single query never oversubscribes the search threads or steals pool
     * capacity from concurrent queries under load. Without that context (e.g. a bare {@link IndexSearcher} in a
     * microbenchmark) there is no pool to consult, so fall back to the host CPU count.
     */
    private static int scoringParallelism(IndexSearcher indexSearcher) {
        if (WORKER_OVERRIDE >= 1) {
            return WORKER_OVERRIDE;
        }
        if (indexSearcher instanceof ContextIndexSearcher contextIndexSearcher) {
            return Math.max(1, contextIndexSearcher.getMaximumNumberOfSlices());
        }
        return Math.max(1, Runtime.getRuntime().availableProcessors());
    }

    @Override
    public void visit(QueryVisitor visitor) {
        if (visitor.acceptField(field)) {
            visitor.visitLeaf(this);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        AbstractIVFKnnVectorQuery that = (AbstractIVFKnnVectorQuery) o;
        return k == that.k
            && numCands == that.numCands
            && Objects.equals(field, that.field)
            && Objects.equals(filter, that.filter)
            && Objects.equals(providedVisitRatio, that.providedVisitRatio);
    }

    @Override
    public int hashCode() {
        return Objects.hash(field, k, numCands, filter, providedVisitRatio);
    }

    @Override
    public Query rewrite(IndexSearcher indexSearcher) throws IOException {
        vectorOpsCount = 0;
        IndexReader reader = indexSearcher.getIndexReader();

        final Weight filterWeight;
        if (filter != null) {
            BooleanQuery booleanQuery = new BooleanQuery.Builder().add(filter, BooleanClause.Occur.FILTER)
                .add(new FieldExistsQuery(field), BooleanClause.Occur.FILTER)
                .build();
            Query rewritten = indexSearcher.rewrite(booleanQuery);
            if (rewritten.getClass() == MatchNoDocsQuery.class) {
                return rewritten;
            }
            filterWeight = indexSearcher.createWeight(rewritten, ScoreMode.COMPLETE_NO_SCORES, 1f);
        } else {
            filterWeight = null;
        }

        final List<LeafReaderContext> contexts = reader.leaves();
        final IVFCollectorManager knnCollectorManager = getKnnCollectorManager(Math.round(2f * k), indexSearcher);
        final TaskExecutor taskExecutor = indexSearcher.getTaskExecutor();

        if (doPrecondition) {
            for (LeafReaderContext context : contexts) {
                preconditionQuery(context);
            }
        }
        final float[] queryVector = getQueryVector();

        final int parallelism = scoringParallelism(indexSearcher);
        TopDocs topK = rewriteGlobalQueue(contexts, filterWeight, knnCollectorManager, queryVector, taskExecutor, parallelism);
        vectorOpsCount = (int) topK.totalHits.value();
        if (topK.scoreDocs.length == 0) {
            return Queries.NO_DOCS_INSTANCE;
        }
        return new KnnScoreDocQuery(topK.scoreDocs, reader);
    }

    /**
     * Global posting queue path: Phase A selects postings across all leaves, Phase B scores them via a
     * shared worker pool, Phase C merges and deduplicates.
     */
    private TopDocs rewriteGlobalQueue(
        List<LeafReaderContext> contexts,
        Weight filterWeight,
        IVFCollectorManager knnCollectorManager,
        float[] queryVector,
        TaskExecutor taskExecutor,
        int parallelism
    ) throws IOException {
        final List<Callable<LeafSearchTask>> prepTasks = new ArrayList<>(contexts.size());
        for (LeafReaderContext context : contexts) {
            final int leafIdx = context.ord;
            prepTasks.add(() -> prepareLeaf(leafIdx, context, filterWeight, queryVector));
        }
        final List<LeafSearchTask> prepared = taskExecutor.invokeAll(prepTasks);

        final List<LeafSearchTask> ivfLeaves = new ArrayList<>();
        final List<LeafSearchTask> fallbackLeaves = new ArrayList<>();
        for (LeafSearchTask task : prepared) {
            if (task == null) {
                continue;
            }
            if (task.needsFallbackSearch()) {
                fallbackLeaves.add(task);
            } else {
                ivfLeaves.add(task);
            }
        }

        final List<LeafResult> leafResults = new ArrayList<>();
        if (ivfLeaves.isEmpty() == false) {
            if (requiresWholeLeafSlices()) {
                leafResults.addAll(scoreWholeLeaves(ivfLeaves, knnCollectorManager, queryVector, taskExecutor, parallelism));
            } else if (ivfLeaves.size() >= parallelism) {
                // Enough segments to already saturate the worker pool: score each leaf on a single worker, skipping
                // the global flatten + sort + cross-leaf dedup that only earn their keep when one segment must be
                // scored by many workers (the force-merged single-segment case handled by scoreGlobalQueue below).
                leafResults.addAll(scoreFusedPerLeaf(ivfLeaves, knnCollectorManager, queryVector, taskExecutor));
            } else {
                leafResults.addAll(scoreGlobalQueue(ivfLeaves, knnCollectorManager, queryVector, taskExecutor, parallelism));
            }
        }
        if (fallbackLeaves.isEmpty() == false) {
            leafResults.addAll(scoreFallbackLeaves(fallbackLeaves, knnCollectorManager, queryVector, taskExecutor));
        }

        if (leafResults.isEmpty()) {
            return NO_RESULTS;
        }
        return mergeAllResults(leafResults, contexts);
    }

    /**
     * Phase A for a single leaf: build the accept-docs, resolve the IVF reader, and select the closest
     * postings up to a per-leaf budget. Returns {@code null} to skip the leaf (no matching docs / no
     * vectors), a fallback task for non-IVF (e.g. HNSW) leaves, or a populated IVF task otherwise.
     */
    private LeafSearchTask prepareLeaf(int leafIdx, LeafReaderContext ctx, Weight filterWeight, float[] queryVector) throws IOException {
        ESAcceptDocs acceptDocs = buildLeafAcceptDocs(ctx, filterWeight);
        if (acceptDocs == null) {
            return null;
        }
        // Force-materialize the filter bitset and approximate cost now, on this single Phase-A thread, so the
        // Phase-B workers that share this leaf observe an immutable, already-built bitset (no concurrent build).
        acceptDocs.bits();
        acceptDocs.approximateCost();
        if (acceptDocs.sliceOrd() >= 0) {
            // Sliced queries lazily consume a DocValuesSkipper to resolve the slice's doc range. Force it here, on
            // the thread that acquired the skipper in buildLeafAcceptDocs, since skippers are thread-confined and the
            // scoring workers run on other threads.
            acceptDocs.sliceAcceptDocs();
        }

        FieldInfo fieldInfo = ctx.reader().getFieldInfos().fieldInfo(field);
        if (fieldInfo == null || fieldInfo.getVectorDimension() == 0) {
            return null;
        }
        IVFVectorsReader<?> ivfReader = resolveIVFReader(ctx, field);
        if (ivfReader == null || fieldInfo.getVectorEncoding() != VectorEncoding.FLOAT32) {
            return LeafSearchTask.forFallback(leafIdx, ctx, acceptDocs);
        }
        return selectPostings(
            leafIdx,
            ctx,
            ivfReader,
            fieldInfo,
            field,
            acceptDocs,
            queryVector,
            providedVisitRatio,
            numCands,
            strategyK(),
            Math.round(2f * k)
        );
    }

    /**
     * Wildcard capture-helper that performs the scoring-free centroid selection. Binding the reader's
     * {@code E} type parameter here lets the query layer (which only holds {@code IVFVectorsReader<?>})
     * call the {@code E}-typed codec hooks. The budget mirrors {@link IVFVectorsReader#search}: it scales
     * the visit ratio with filter density and ensures enough candidate vectors to fill the oversampled
     * collector. Because Phase A does not score, this is an approximation — the primary recall-parity risk.
     */
    private static <E extends IVFVectorsReader.FieldEntry> LeafSearchTask selectPostings(
        int leafIdx,
        LeafReaderContext ctx,
        IVFVectorsReader<E> reader,
        FieldInfo fieldInfo,
        String field,
        ESAcceptDocs acceptDocs,
        float[] queryVector,
        float providedVisitRatio,
        int numCands,
        int strategyK,
        int collectorK
    ) throws IOException {
        E entry = reader.fieldEntry(fieldInfo.number);
        if (entry == null) {
            return null;
        }
        // Mirror IVFVectorsReader#search: validate the query dimension before touching the codec scoring path,
        // so a mismatch surfaces as an IllegalArgumentException rather than a deep assertion inside the scorer.
        if (fieldInfo.getVectorDimension() != queryVector.length) {
            throw new IllegalArgumentException(
                "vector query dimension: " + queryVector.length + " differs from field dimension: " + fieldInfo.getVectorDimension()
            );
        }
        FloatVectorValues values = reader.getFloatVectorValues(field);
        IndexInput centroids = entry.centroidSlice(reader.ivfCentroids());
        int numVectors = reader.getNumberOfVectors(entry, values, centroids, acceptDocs);
        if (numVectors == 0) {
            return null;
        }
        final float approximateCost = acceptDocs instanceof ESAcceptDocs.ESAcceptDocsAll ? numVectors : acceptDocs.approximateCost();
        float percentFiltered = Math.max(0f, Math.min(1f, approximateCost / numVectors));
        float effRatio = reader.effectiveVisitRatio(providedVisitRatio, numCands, strategyK, numVectors);

        // We account for SOAR vectors here: a vector can be visited twice, so we multiply by 2.
        long maxVectorVisited = Math.max(1L, (long) Math.ceil(2.0 * effRatio * numVectors));
        if (percentFiltered > 0f && percentFiltered < 0.5f) {
            // Scale the budget up under selective filters so enough surviving vectors are visited.
            double scale = 0.5 / Math.max(percentFiltered, 0.25);
            maxVectorVisited = Math.min((long) numVectors, Math.round(maxVectorVisited * scale));
        }
        long requiredToFill = percentFiltered > 0f ? (long) Math.ceil(collectorK / (double) percentFiltered) : collectorK;
        requiredToFill = Math.min(requiredToFill, numVectors);
        long targetVectors = Math.max(maxVectorVisited, requiredToFill);
        // When the budget reaches the whole segment (typically under a restrictive filter, where the
        // non-parallel search would scan to exhaustion to fill the collector), visit every posting. Otherwise
        // estimatePostingVectorCount overshooting targetVectors could stop selection one or more postings short
        // and drop matching docs that live in the tail.
        final boolean visitAllPostings = targetVectors >= numVectors;

        IndexInput postingSlice = entry.postingListSlice(reader.ivfClusters());
        CentroidIterator centroidIterator = reader.getCentroidIterator(
            fieldInfo,
            entry.numCentroids(),
            centroids,
            queryVector,
            postingSlice,
            acceptDocs,
            approximateCost,
            values,
            effRatio
        );
        List<PostingSearchTask> postings = new ArrayList<>();
        long expectedDocs = 0;
        while (centroidIterator.hasNext() && (visitAllPostings || expectedDocs < targetVectors)) {
            PostingMetadata metadata = centroidIterator.nextPosting();
            int estimated = reader.estimatePostingVectorCount(entry, fieldInfo, metadata);
            postings.add(new PostingSearchTask(metadata.queryCentroidOrdinal(), metadata.documentCentroidScore(), estimated, metadata));
            expectedDocs += estimated;
        }
        // Filter back-fill pre-compensation (mirrors IVFVectorsReader#search Loop 2).
        // When a filter is present, the primary budget may yield too few post-filter vectors.
        // Estimate actualDocs and select more postings until the expected post-filter count
        // meets the back-fill target. Phase B's competitive frontier prunes these if the
        // collector fills from the primary set alone.
        if (percentFiltered < 1.0f && !visitAllPostings) {
            float unfilteredRatioVisited = (float) expectedDocs / numVectors;
            long filteredVectors = (long) Math.ceil(numVectors * (double) percentFiltered);
            float expectedScored = Math.min(2.0f * filteredVectors * unfilteredRatioVisited, expectedDocs / 2.0f);
            float estimatedActualDocs = expectedDocs * percentFiltered;

            while (centroidIterator.hasNext() && (estimatedActualDocs < expectedScored || estimatedActualDocs < collectorK)) {
                PostingMetadata metadata = centroidIterator.nextPosting();
                int estimated = reader.estimatePostingVectorCount(entry, fieldInfo, metadata);
                postings.add(new PostingSearchTask(metadata.queryCentroidOrdinal(), metadata.documentCentroidScore(), estimated, metadata));
                expectedDocs += estimated;
                estimatedActualDocs += estimated * percentFiltered;
            }
        }
        if (postings.isEmpty()) {
            return null;
        }
        return LeafSearchTask.forIvf(
            leafIdx,
            ctx,
            reader,
            entry,
            fieldInfo,
            acceptDocs,
            postings.toArray(new PostingSearchTask[0]),
            effRatio,
            numCands
        );
    }

    private static IVFVectorsReader<?> resolveIVFReader(LeafReaderContext ctx, String field) throws IOException {
        SegmentReader segmentReader = Lucene.tryUnwrapSegmentReader(ctx.reader());
        if (segmentReader == null) {
            return null;
        }
        KnnVectorsReader vectorsReader = segmentReader.getVectorReader();
        if (vectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader perFieldReader) {
            vectorsReader = perFieldReader.getFieldReader(field);
        }
        return vectorsReader instanceof IVFVectorsReader<?> ivfReader ? ivfReader : null;
    }

    /**
     * Phase B (default): flatten every selected posting from every IVF leaf into a single closest-first
     * ordering and let a fixed worker pool drain it via one lock-free cursor. Workers score through their
     * own thread-confined per-leaf state and share only the competitive frontier.
     */
    private List<LeafResult> scoreGlobalQueue(
        List<LeafSearchTask> ivfLeaves,
        IVFCollectorManager knnCollectorManager,
        float[] queryVector,
        TaskExecutor taskExecutor,
        int parallelism
    ) throws IOException {
        int total = 0;
        for (LeafSearchTask leaf : ivfLeaves) {
            total += leaf.postingsClosestFirst().length;
        }
        final GlobalPosting[] globalPostings = new GlobalPosting[total];
        int gi = 0;
        for (LeafSearchTask leaf : ivfLeaves) {
            for (PostingSearchTask posting : leaf.postingsClosestFirst()) {
                globalPostings[gi++] = new GlobalPosting(
                    leaf.leafIdx(),
                    posting.metadata(),
                    posting.score(),
                    posting.estimatedVectorCount()
                );
            }
        }
        // Single closest-first ordering: this is the queue. Score the globally-closest centroid's posting next.
        Arrays.sort(globalPostings, (a, b) -> Float.compare(b.centroidScore(), a.centroidScore()));

        final IntObjectHashMap<LeafSearchTask> leafById = new IntObjectHashMap<>();
        for (LeafSearchTask leaf : ivfLeaves) {
            leafById.put(leaf.leafIdx(), leaf);
        }

        final int workerCount = Math.min(Math.max(1, parallelism), globalPostings.length);
        final AtomicInteger cursor = new AtomicInteger(0);
        final List<Callable<List<LeafResult>>> workers = new ArrayList<>(workerCount);
        for (int w = 0; w < workerCount; w++) {
            workers.add(() -> runWorker(globalPostings, cursor, leafById, knnCollectorManager, queryVector, workerCount));
        }
        final List<LeafResult> all = new ArrayList<>();
        for (List<LeafResult> workerResults : taskExecutor.invokeAll(workers)) {
            all.addAll(workerResults);
        }
        return all;
    }

    private List<LeafResult> runWorker(
        GlobalPosting[] globalPostings,
        AtomicInteger cursor,
        IntObjectHashMap<LeafSearchTask> leafById,
        IVFCollectorManager knnCollectorManager,
        float[] queryVector,
        int workerCount
    ) throws IOException {
        final IntObjectHashMap<LeafWorkerState> states = new IntObjectHashMap<>();
        try {
            // Claim a contiguous chunk of the global closest-first queue per CAS so the worker owns a run it can
            // prefetch ahead of scoring (overlapping I/O with CPU). The chunk is kept small so the cursor still
            // hands out far more chunks than workers, preserving the fine-grained load balancing of the queue.
            final int chunk = chunkSize(globalPostings.length, workerCount);
            int start;
            while ((start = cursor.getAndAdd(chunk)) < globalPostings.length) {
                final int end = Math.min(start + chunk, globalPostings.length);
                // Sliding-window prefetch: rather than paging in the whole chunk up front (which, for a large run,
                // lets the earliest pages be evicted before they are scored), keep a bounded window of postings in
                // flight. Prime the window, then prefetch the posting PREFETCH_DISTANCE ahead just before scoring the
                // current one so each posting's I/O overlaps with scoring of the ones in front of it.
                final int distance = Math.min(PREFETCH_DISTANCE, end - start);
                for (int i = start; i < start + distance; i++) {
                    prefetchGlobalPosting(globalPostings[i], states, leafById, knnCollectorManager, queryVector, workerCount);
                }
                for (int i = start; i < end; i++) {
                    final int ahead = i + distance;
                    if (ahead < end) {
                        prefetchGlobalPosting(globalPostings[ahead], states, leafById, knnCollectorManager, queryVector, workerCount);
                    }
                    GlobalPosting posting = globalPostings[i];
                    LeafWorkerState state = states.get(posting.leafIdx());
                    if (state == null) {
                        continue;
                    }
                    state.visitor.resetPostingsScorer(posting.metadata());
                    state.visitor.visit(state.collector);
                    // Publish/refresh the shared competitive frontier after each posting.
                    state.strategy.nextVectorsBlock();
                }
            }
            final List<LeafResult> results = new ArrayList<>(states.size());
            for (IntObjectHashMap.IntObjectCursor<LeafWorkerState> entry : states) {
                if (entry.value == null) {
                    continue;
                }
                TopDocs topDocs = drain(entry.value.collector);
                if (topDocs.scoreDocs.length > 0) {
                    results.add(new LeafResult(entry.key, topDocs));
                }
            }
            return results;
        } finally {
            closeStates(states);
        }
    }

    /**
     * Ensures this worker holds state for the posting's leaf and issues a prefetch for the posting's bytes. Used to
     * fill the sliding prefetch window in {@link #runWorker}; building state here (rather than only at scoring time)
     * is required so the confined/cloned handle exists before its I/O is hinted.
     */
    private void prefetchGlobalPosting(
        GlobalPosting posting,
        IntObjectHashMap<LeafWorkerState> states,
        IntObjectHashMap<LeafSearchTask> leafById,
        IVFCollectorManager knnCollectorManager,
        float[] queryVector,
        int workerCount
    ) throws IOException {
        LeafWorkerState state = ensureState(states, posting.leafIdx(), leafById, knnCollectorManager, queryVector, workerCount);
        if (state != null) {
            state.visitor.prefetch(posting.metadata());
        }
    }

    /**
     * Lazily builds (and caches) this worker's thread-confined state for {@code leafIdx}. A leaf with more than one
     * selected posting can have those postings claimed by several workers, so it is marked contended and scored
     * through a confined mapping; a single-posting leaf (or a single-worker query) just clones the shared handle.
     * A {@code null} state (collector declined) is cached too so it is not rebuilt for every posting of that leaf.
     */
    private LeafWorkerState ensureState(
        IntObjectHashMap<LeafWorkerState> states,
        int leafIdx,
        IntObjectHashMap<LeafSearchTask> leafById,
        IVFCollectorManager knnCollectorManager,
        float[] queryVector,
        int workerCount
    ) throws IOException {
        if (states.containsKey(leafIdx)) {
            return states.get(leafIdx);
        }
        LeafSearchTask leaf = leafById.get(leafIdx);
        boolean contended = workerCount > 1 && leaf.postingsClosestFirst().length > 1;
        LeafWorkerState state = buildState(leaf, knnCollectorManager, queryVector, contended);
        states.put(leafIdx, state);
        return state;
    }

    /**
     * Per-chunk claim size for the global queue: small enough that there remain far more chunks than workers
     * (so load stays balanced via the shared cursor), but {@code > 1} so each claim covers a run a worker can
     * prefetch ahead of scoring. Falls back to {@code 1} when there are too few postings to chunk.
     */
    static int chunkSize(int totalPostings, int workerCount) {
        if (totalPostings <= workerCount || workerCount <= 1) {
            return 1;
        }
        int targetChunks = workerCount * QUEUE_CHUNK_OVERSUBSCRIPTION;
        return Math.max(1, totalPostings / targetChunks);
    }

    private static void closeStates(IntObjectHashMap<LeafWorkerState> states) throws IOException {
        final List<LeafWorkerState> toClose = new ArrayList<>(states.size());
        for (IntObjectHashMap.IntObjectCursor<LeafWorkerState> entry : states) {
            if (entry.value != null) {
                toClose.add(entry.value);
            }
        }
        IOUtils.close(toClose);
    }

    /**
     * Phase B for diversifying (block-join) queries: the work items are whole leaves, ordered closest-first
     * by their best centroid score. A single worker scores an entire leaf into one collector built from that
     * leaf's real context, so the parent bitset resolves and at most one child per parent survives — two
     * children of the same parent can never end up in separate worker collectors.
     */
    private List<LeafResult> scoreWholeLeaves(
        List<LeafSearchTask> ivfLeaves,
        IVFCollectorManager knnCollectorManager,
        float[] queryVector,
        TaskExecutor taskExecutor,
        int parallelism
    ) throws IOException {
        final LeafSearchTask[] ordered = ivfLeaves.toArray(new LeafSearchTask[0]);
        Arrays.sort(ordered, (a, b) -> Float.compare(b.bestCentroidScore(), a.bestCentroidScore()));

        final int workerCount = Math.min(Math.max(1, parallelism), ordered.length);
        final AtomicInteger cursor = new AtomicInteger(0);
        final List<Callable<List<LeafResult>>> workers = new ArrayList<>(workerCount);
        for (int w = 0; w < workerCount; w++) {
            workers.add(() -> runWholeLeafWorker(ordered, cursor, knnCollectorManager, queryVector));
        }
        final List<LeafResult> all = new ArrayList<>();
        for (List<LeafResult> workerResults : taskExecutor.invokeAll(workers)) {
            all.addAll(workerResults);
        }
        return all;
    }

    private List<LeafResult> runWholeLeafWorker(
        LeafSearchTask[] ordered,
        AtomicInteger cursor,
        IVFCollectorManager knnCollectorManager,
        float[] queryVector
    ) throws IOException {
        final List<LeafResult> results = new ArrayList<>();
        int idx;
        while ((idx = cursor.getAndIncrement()) < ordered.length) {
            LeafResult result = scoreLeafFully(ordered[idx], knnCollectorManager, queryVector);
            if (result != null) {
                results.add(result);
            }
        }
        return results;
    }

    /**
     * Fused fast path for when there are at least as many IVF leaves as workers ({@code leaves >= parallelism}): the
     * segments already saturate the worker pool, so score each leaf on a single worker (one task per leaf) and skip
     * the global flatten + closest-first sort + cross-leaf dedup that only pay off when a single segment must be
     * scored by many workers. Each leaf still feeds the shared competitive frontier, so a leaf that finishes early
     * raises the global bar for the leaves still scoring.
     */
    private List<LeafResult> scoreFusedPerLeaf(
        List<LeafSearchTask> ivfLeaves,
        IVFCollectorManager knnCollectorManager,
        float[] queryVector,
        TaskExecutor taskExecutor
    ) throws IOException {
        final List<Callable<LeafResult>> tasks = new ArrayList<>(ivfLeaves.size());
        for (LeafSearchTask leaf : ivfLeaves) {
            tasks.add(() -> scoreLeafFully(leaf, knnCollectorManager, queryVector));
        }
        final List<LeafResult> results = new ArrayList<>(ivfLeaves.size());
        for (LeafResult result : taskExecutor.invokeAll(tasks)) {
            if (result != null) {
                results.add(result);
            }
        }
        return results;
    }

    /**
     * Scores all of a leaf's selected postings on the calling worker thread into a single collector and drains the
     * result. The leaf is scored start-to-finish by one thread, so it is never contended: a plain clone of the
     * shared handles suffices (see {@link #buildState}). Postings are prefetched before scoring so the leaf's I/O
     * overlaps with its CPU work. Returns {@code null} when the leaf yields no collector or no hits.
     */
    private LeafResult scoreLeafFully(LeafSearchTask leaf, IVFCollectorManager knnCollectorManager, float[] queryVector)
        throws IOException {
        try (LeafWorkerState state = buildState(leaf, knnCollectorManager, queryVector, false)) {
            if (state == null) {
                return null;
            }
            // Sliding-window prefetch over the leaf's postings: keep at most PREFETCH_DISTANCE prefetches in flight
            // instead of hinting the whole (potentially very large, on a force-merged single segment) leaf at once.
            final PostingSearchTask[] postings = leaf.postingsClosestFirst();
            final int distance = Math.min(PREFETCH_DISTANCE, postings.length);
            for (int i = 0; i < distance; i++) {
                state.visitor.prefetch(postings[i].metadata());
            }
            for (int i = 0; i < postings.length; i++) {
                final int ahead = i + distance;
                if (ahead < postings.length) {
                    state.visitor.prefetch(postings[ahead].metadata());
                }
                state.visitor.resetPostingsScorer(postings[i].metadata());
                state.visitor.visit(state.collector);
                state.strategy.nextVectorsBlock();
            }
            TopDocs topDocs = drain(state.collector);
            return topDocs.scoreDocs.length > 0 ? new LeafResult(leaf.leafIdx(), topDocs) : null;
        }
    }

    /**
     * Builds the per-worker scoring state for one leaf. A {@code contended} leaf (one whose postings may be scored
     * concurrently by more than one worker) gets thread-confined cluster/centroid mappings via
     * {@link IVFVectorsReader#openLightweightClusters()} so reads avoid the cross-thread compare-and-swap on the
     * shared {@code MemorySegment} session. A leaf scored by a single worker instead {@link IndexInput#clone()
     * clones} the shared handles — clones share the same virtual address mappings (warm TLB entries) while giving
     * each worker its own seek position. The returned state owns its handles and must be closed on this thread.
     * Returns {@code null} (after releasing any opened handle) when the collector manager declines the leaf (e.g. a
     * diversifying join with no parent bitset).
     */
    private LeafWorkerState buildState(LeafSearchTask leaf, IVFCollectorManager knnCollectorManager, float[] queryVector, boolean contended)
        throws IOException {
        final IVFVectorsReader<?> reader = leaf.reader();
        final IVFVectorsReader.FieldEntry entry = leaf.entry();
        IndexInput clustersHandle = null;
        IndexInput centroidsHandle = null;
        try {
            if (contended) {
                clustersHandle = reader.openLightweightClusters();
                centroidsHandle = reader.openLightweightCentroids();
            } else {
                clustersHandle = reader.ivfClusters().clone();
                centroidsHandle = reader.ivfCentroids().clone();
            }
            IndexInput postingListSlice = entry.postingListSlice(clustersHandle);
            IndexInput centroidSlice = entry.centroidSlice(centroidsHandle);
            FloatVectorValues values = reader.getFloatVectorValues(field);
            IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(
                leaf.effectiveVisitRatio(),
                leaf.numCands(),
                strategyK(),
                knnCollectorManager.longAccumulator
            );
            AbstractMaxScoreKnnCollector collector = knnCollectorManager.newCollector(Integer.MAX_VALUE, strategy, leaf.ctx());
            if (collector == null) {
                IOUtils.close(clustersHandle, centroidsHandle);
                return null;
            }
            strategy.setCollector(collector);
            IVFVectorsReader.PostingVisitor visitor = reader.getPostingVisitor(
                leaf.fieldInfo(),
                values,
                postingListSlice,
                queryVector,
                leaf.acceptDocs().bits(),
                centroidSlice,
                leaf.acceptDocs()
            );
            return new LeafWorkerState(clustersHandle, centroidsHandle, visitor, collector, strategy);
        } catch (Throwable t) {
            IOUtils.closeWhileHandlingException(clustersHandle, centroidsHandle);
            throw t;
        }
    }

    /**
     * Scores non-IVF leaves (e.g. HNSW segments in a mixed index) the way the non-parallel path did:
     * one task per leaf delegating to {@link LeafReader#searchNearestVectors}. These leaves share the same
     * collector manager and competitive frontier as the IVF posting workers.
     */
    private List<LeafResult> scoreFallbackLeaves(
        List<LeafSearchTask> fallbackLeaves,
        IVFCollectorManager knnCollectorManager,
        float[] queryVector,
        TaskExecutor taskExecutor
    ) throws IOException {
        final List<Callable<LeafResult>> tasks = new ArrayList<>(fallbackLeaves.size());
        for (LeafSearchTask leaf : fallbackLeaves) {
            tasks.add(() -> runFallback(leaf, knnCollectorManager, queryVector));
        }
        final List<LeafResult> results = new ArrayList<>();
        for (LeafResult result : taskExecutor.invokeAll(tasks)) {
            if (result != null) {
                results.add(result);
            }
        }
        return results;
    }

    private LeafResult runFallback(LeafSearchTask leaf, IVFCollectorManager knnCollectorManager, float[] queryVector) throws IOException {
        IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(
            providedVisitRatio,
            numCands,
            strategyK(),
            knnCollectorManager.longAccumulator
        );
        AbstractMaxScoreKnnCollector collector = knnCollectorManager.newCollector(Integer.MAX_VALUE, strategy, leaf.ctx());
        if (collector == null) {
            return null;
        }
        strategy.setCollector(collector);
        leaf.ctx().reader().searchNearestVectors(field, queryVector, collector, leaf.acceptDocs());
        TopDocs topDocs = drain(collector);
        return topDocs.scoreDocs.length > 0 ? new LeafResult(leaf.leafIdx(), topDocs) : null;
    }

    private static TopDocs drain(AbstractMaxScoreKnnCollector collector) {
        TopDocs topDocs = collector instanceof BulkKnnCollector bulkKnnCollector ? bulkKnnCollector.unsortedTopK() : collector.topDocs();
        return topDocs != null ? topDocs : NO_RESULTS;
    }

    /**
     * Phase C: group the per-(worker, leaf) results by leaf, dedup by global doc id keeping the highest
     * score (a SOAR vector scored from two postings by two workers may surface twice), then merge across
     * leaves down to {@code k}.
     */
    private TopDocs mergeAllResults(List<LeafResult> leafResults, List<LeafReaderContext> contexts) {
        final IntObjectHashMap<List<TopDocs>> byLeaf = new IntObjectHashMap<>();
        for (LeafResult leafResult : leafResults) {
            List<TopDocs> partials = byLeaf.get(leafResult.leafIdx());
            if (partials == null) {
                partials = new ArrayList<>();
                byLeaf.put(leafResult.leafIdx(), partials);
            }
            partials.add(leafResult.topDocs());
        }
        final List<TopDocs> perLeaf = new ArrayList<>(byLeaf.size());
        for (IntObjectHashMap.IntObjectCursor<List<TopDocs>> cursor : byLeaf) {
            int docBase = contexts.get(cursor.key).docBase;
            perLeaf.add(mergeLeafPartials(cursor.value, docBase));
        }
        return mergeLeafResults(k, perLeaf.toArray(TopDocs[]::new));
    }

    private static TopDocs mergeLeafPartials(List<TopDocs> partials, int docBase) {
        final IntObjectHashMap<ScoreDoc> bestByGlobalDoc = new IntObjectHashMap<>();
        long totalHitsValue = 0;
        TotalHits.Relation relation = TotalHits.Relation.EQUAL_TO;
        for (TopDocs partial : partials) {
            totalHitsValue += partial.totalHits.value();
            if (partial.totalHits.relation() == TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO) {
                relation = TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO;
            }
            for (ScoreDoc scoreDoc : partial.scoreDocs) {
                int globalDoc = scoreDoc.doc + docBase;
                ScoreDoc existing = bestByGlobalDoc.get(globalDoc);
                if (existing == null || scoreDoc.score > existing.score) {
                    scoreDoc.doc = globalDoc;
                    bestByGlobalDoc.put(globalDoc, scoreDoc);
                }
            }
        }
        ScoreDoc[] deduplicated = new ScoreDoc[bestByGlobalDoc.size()];
        int i = 0;
        for (IntObjectHashMap.IntObjectCursor<ScoreDoc> cursor : bestByGlobalDoc) {
            deduplicated[i++] = cursor.value;
        }
        return new TopDocs(new TotalHits(totalHitsValue, relation), deduplicated);
    }

    private TopDocs mergeLeafResults(int mergeK, TopDocs[] perLeafResults) {
        // During merge across segments, always favor bulk pivot collection.
        // Segment-level unsorted gathering avoids per-segment sorting work.
        BulkNeighborQueue mergeQueue = BulkNeighborQueue.forMerging(mergeK);
        long totalHitsValue = 0;
        TotalHits.Relation relation = TotalHits.Relation.EQUAL_TO;
        for (TopDocs topDocs : perLeafResults) {
            totalHitsValue += topDocs.totalHits.value();
            if (topDocs.totalHits.relation() == TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO) {
                relation = TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO;
            }
            if (topDocs.scoreDocs.length == 0) {
                continue;
            }
            int count = topDocs.scoreDocs.length;
            int[] docs = new int[count];
            float[] scores = new float[count];
            float bestScore = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < count; i++) {
                ScoreDoc scoreDoc = topDocs.scoreDocs[i];
                docs[i] = scoreDoc.doc;
                scores[i] = scoreDoc.score;
                if (scoreDoc.score > bestScore) {
                    bestScore = scoreDoc.score;
                }
            }
            mergeQueue.insertWithOverflowBulk(docs, scores, count, bestScore);
        }
        ScoreDoc[] mergedScoreDocs = new ScoreDoc[mergeQueue.size()];
        int[] index = new int[] { mergedScoreDocs.length - 1 };
        mergeQueue.drain(
            encoded -> mergedScoreDocs[index[0]--] = new ScoreDoc(mergeQueue.decodeNodeId(encoded), mergeQueue.decodeScore(encoded))
        );
        return new TopDocs(new TotalHits(totalHitsValue, relation), mergedScoreDocs);
    }

    /**
     * Preconditions (transforms) the query vector against the leaf's preconditioner, if any. Mutates shared
     * query state, so the pipeline invokes it sequentially before the parallel phases.
     */
    abstract void preconditionQuery(LeafReaderContext context) throws IOException;

    /** The (possibly preconditioned) query vector to score against, resolved after preconditioning runs. */
    protected abstract float[] getQueryVector();

    /**
     * The {@code k} fed to the per-worker search strategy for competitive-frontier math. Defaults to
     * {@link #k}; oversampling queries override this to return the pre-oversample k.
     */
    protected int strategyK() {
        return k;
    }

    /**
     * Whether scoring must keep each leaf whole (one collector per leaf) instead of distributing a leaf's
     * postings across workers. Diversifying (block-join) queries require this so a parent's children are not
     * split across independent worker collectors. Defaults to {@code false}.
     */
    protected boolean requiresWholeLeafSlices() {
        return false;
    }

    /**
     * Builds the {@link ESAcceptDocs} for a leaf from its live docs and the optional filter, returning
     * {@code null} to skip the leaf entirely (e.g. the filter matches nothing here). Sliced queries override
     * this to additionally restrict scoring to the slice's contiguous doc range.
     */
    protected ESAcceptDocs buildLeafAcceptDocs(LeafReaderContext ctx, Weight filterWeight) throws IOException {
        final LeafReader leafReader = ctx.reader();
        final Bits liveDocs = leafReader.getLiveDocs();
        final int maxDoc = leafReader.maxDoc();
        if (filterWeight == null) {
            return liveDocs == null ? new ESAcceptDocs.ESAcceptDocsAll() : new ESAcceptDocs.BitsAcceptDocs(liveDocs, maxDoc);
        }
        ScorerSupplier supplier = filterWeight.scorerSupplier(ctx);
        if (supplier == null) {
            return null;
        }
        return new ESAcceptDocs.ScorerSupplierAcceptDocs(supplier, liveDocs, maxDoc);
    }

    protected IVFCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
        return new IVFCollectorManager(k, searcher);
    }

    @Override
    public final void profile(QueryProfiler queryProfiler) {
        queryProfiler.addVectorOpsCount(vectorOpsCount);
    }

    /** A selected posting list within a single leaf, ordered closest-first by {@link #score}. */
    record PostingSearchTask(int centroidOrd, float score, int estimatedVectorCount, PostingMetadata metadata) {}

    /** A selected posting list lifted into the global, cross-leaf closest-first ordering. */
    record GlobalPosting(int leafIdx, PostingMetadata metadata, float centroidScore, int estVectorCount) {}

    /** Per-(worker, leaf) partial result drained from a worker's collector; doc ids are still leaf-local. */
    record LeafResult(int leafIdx, TopDocs topDocs) {}

    /**
     * Phase A output for one leaf. IVF leaves carry the selected postings and the resolved per-leaf budget;
     * non-IVF leaves carry only {@code needsFallbackSearch=true} and their accept-docs.
     */
    record LeafSearchTask(
        int leafIdx,
        LeafReaderContext ctx,
        IVFVectorsReader<?> reader,
        IVFVectorsReader.FieldEntry entry,
        FieldInfo fieldInfo,
        ESAcceptDocs acceptDocs,
        PostingSearchTask[] postingsClosestFirst,
        float effectiveVisitRatio,
        int numCands,
        boolean needsFallbackSearch
    ) {
        static LeafSearchTask forIvf(
            int leafIdx,
            LeafReaderContext ctx,
            IVFVectorsReader<?> reader,
            IVFVectorsReader.FieldEntry entry,
            FieldInfo fieldInfo,
            ESAcceptDocs acceptDocs,
            PostingSearchTask[] postingsClosestFirst,
            float effectiveVisitRatio,
            int numCands
        ) {
            return new LeafSearchTask(
                leafIdx,
                ctx,
                reader,
                entry,
                fieldInfo,
                acceptDocs,
                postingsClosestFirst,
                effectiveVisitRatio,
                numCands,
                false
            );
        }

        static LeafSearchTask forFallback(int leafIdx, LeafReaderContext ctx, ESAcceptDocs acceptDocs) {
            return new LeafSearchTask(leafIdx, ctx, null, null, null, acceptDocs, null, 0f, 0, true);
        }

        float bestCentroidScore() {
            float best = Float.NEGATIVE_INFINITY;
            for (PostingSearchTask posting : postingsClosestFirst) {
                if (posting.score() > best) {
                    best = posting.score();
                }
            }
            return best;
        }
    }

    static class IVFCollectorManager implements KnnCollectorManager {
        private final int k;
        final LongAccumulator longAccumulator;

        IVFCollectorManager(int k, IndexSearcher searcher) {
            this.k = k;
            // Posting-granular parallelism creates concurrency even within a single leaf, so the shared
            // competitive frontier must always exist (the non-parallel path only allocated it for >1 leaf).
            this.longAccumulator = new LongAccumulator(Long::max, LEAST_COMPETITIVE);
        }

        @Override
        public AbstractMaxScoreKnnCollector newCollector(int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context)
            throws IOException {
            return new MaxScoreTopKnnCollector(k, visitedLimit, searchStrategy);
        }
    }
}
