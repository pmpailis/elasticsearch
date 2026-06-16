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
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.search.AcceptDocs;
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

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAccumulator;

import static org.elasticsearch.search.vectors.AbstractMaxScoreKnnCollector.LEAST_COMPETITIVE;

/**
 * Base class for the parallel IVF (DiskBBQ) kNN queries.
 * <p>
 * The query picks one of two scoring strategies based on how many segments there are relative to the available
 * worker parallelism:
 * <ul>
 *   <li><b>Many segments ({@code leaves >= parallelism}):</b> the segments already saturate the worker pool, so
 *       each leaf is scored by a single worker in one fused prepare+score pass — same per-leaf parallelism as a
 *       standard segment search, with no inter-phase barrier and no within-leaf banding. The competitive frontier
 *       is still shared across leaves so the recall benefit is preserved.</li>
 *   <li><b>Few segments ({@code leaves < parallelism}, e.g. a single force-merged segment):</b> scoring is split
 *       into three phases so parallelism reaches posting-list granularity — a single segment is scored by many
 *       workers, not just one thread per leaf:
 *       <ol>
 *         <li><b>Phase A</b> (one task per leaf): resolve the codec reader, materialize the filter, score the query
 *             against centroids and statically select the closest posting lists up to a per-leaf vector budget.</li>
 *         <li><b>Phase B</b> (a fixed worker pool draining a lock-free cursor over pre-sorted {@link Slice}s): each
 *             worker scores posting lists through its own thread-confined state, sharing the single
 *             competitive-frontier accumulator.</li>
 *       </ol>
 *   </li>
 * </ul>
 * Both strategies feed a final merge that dedups globally by document and builds the {@link KnnScoreDocQuery}.
 */
abstract class AbstractIVFKnnVectorQuery extends Query implements QueryProfilerProvider {

    static final TopDocs NO_RESULTS = TopDocsCollector.EMPTY_TOPDOCS;

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

        // we request 2*k as we are using it as an approximation measure: we need at least 2*k results to cover
        // overspill (soar) duplicates and so the shared competitive frontier never skips a true top-k block.
        IVFCollectorManager knnCollectorManager = getKnnCollectorManager(Math.round(2f * k), indexSearcher);
        TaskExecutor taskExecutor = indexSearcher.getTaskExecutor();
        List<LeafReaderContext> leafReaderContexts = reader.leaves();

        // Precondition the query vector once, on this (single) thread, before any leaf is prepared. The
        // transform mutates shared query state and must not race with the parallel phases below.
        if (doPrecondition) {
            for (LeafReaderContext ctx : leafReaderContexts) {
                preconditionQuery(ctx);
            }
        }

        int parallelism = scoringParallelism(indexSearcher);
        List<LeafResult> allResults = new ArrayList<>();

        if (requiresWholeLeafSlices() == false && leafReaderContexts.size() >= parallelism) {
            // Enough segments to already saturate the worker pool: score each leaf on a single worker in one fused
            // prepare+score pass. This mirrors the per-leaf parallelism of a standard segment search — no Phase A/B
            // barrier, no within-leaf banding, and (since each leaf is touched by exactly one thread) a plain clone
            // instead of a thread-confined remap — while still sharing the competitive frontier across leaves so the
            // recall benefit is preserved. Within-leaf slicing only pays off when there are fewer leaves than workers,
            // which is handled by the banding path below.
            List<Callable<List<LeafResult>>> leafTasks = new ArrayList<>(leafReaderContexts.size());
            for (LeafReaderContext ctx : leafReaderContexts) {
                leafTasks.add(() -> scoreLeafFused(ctx, filterWeight, knnCollectorManager));
            }
            for (List<LeafResult> partial : taskExecutor.invokeAll(leafTasks)) {
                allResults.addAll(partial);
            }
        } else {
            // Phase A: prepare each leaf in parallel (centroid scoring + posting selection; no vectors scored).
            List<Callable<LeafSearchTask>> prepTasks = new ArrayList<>(leafReaderContexts.size());
            for (LeafReaderContext ctx : leafReaderContexts) {
                prepTasks.add(() -> prepareLeaf(ctx, filterWeight));
            }
            List<LeafSearchTask> prepared = taskExecutor.invokeAll(prepTasks);

            List<LeafSearchTask> ivfLeaves = new ArrayList<>(prepared.size());
            List<LeafSearchTask> fallbackLeaves = new ArrayList<>();
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

            // Phase B: distribute the selected postings into slices and score them with a fixed worker pool, banding
            // large leaves so a handful of segments can still be scored by many workers (the single force-merged
            // segment case). Split leaves are scored through thread-confined mappings (see markContendedLeaves).
            Slice[] slices = IVFSlicePlanner.buildSlices(ivfLeaves, parallelism, requiresWholeLeafSlices());
            boolean[] contended = IVFSlicePlanner.markContendedLeaves(ivfLeaves, slices);
            AtomicInteger cursor = new AtomicInteger(0);

            int workerCount = slices.length > 0 ? Math.min(parallelism, slices.length) : 0;
            List<Callable<List<LeafResult>>> scoreTasks = new ArrayList<>();
            for (int w = 0; w < workerCount; w++) {
                scoreTasks.add(() -> runWorker(slices, cursor, contended, ivfLeaves, knnCollectorManager));
            }
            for (LeafSearchTask fallback : fallbackLeaves) {
                scoreTasks.add(() -> List.of(runFallback(fallback, knnCollectorManager)));
            }

            if (scoreTasks.isEmpty() == false) {
                for (List<LeafResult> partial : taskExecutor.invokeAll(scoreTasks)) {
                    allResults.addAll(partial);
                }
            }
        }

        // Phase C: merge, dedup globally by document, and build the result query.
        TopDocs topK = mergeResults(k, allResults);
        vectorOpsCount = (int) topK.totalHits.value();
        if (topK.scoreDocs.length == 0) {
            return Queries.NO_DOCS_INSTANCE;
        }
        return new KnnScoreDocQuery(topK.scoreDocs, reader);
    }

    /**
     * The number of Phase-B scoring workers to run. Under a {@link ContextIndexSearcher} (the production search path)
     * this is the searcher's resolved maximum slice count, which is already sized to the search thread pool and falls
     * back to {@code 1} when the pool is saturated — so a single query never oversubscribes the search threads or
     * steals pool capacity from concurrent queries under load. Without that context (e.g. a bare {@link IndexSearcher}
     * in a microbenchmark) there is no pool to consult, so fall back to the host CPU count.
     */
    private static int scoringParallelism(IndexSearcher indexSearcher) {
        if (indexSearcher instanceof ContextIndexSearcher contextIndexSearcher) {
            return Math.max(1, contextIndexSearcher.getMaximumNumberOfSlices());
        }
        return Math.max(1, Runtime.getRuntime().availableProcessors());
    }

    /**
     * Phase A for a single leaf: build the {@link ESAcceptDocs}, warm its lazy caches single-threaded, resolve the
     * field's codec reader, and (for an IVF field) select the closest posting lists. Returns {@code null} when the
     * leaf has nothing to contribute (no matching docs, no vectors, or no postings selected).
     */
    private LeafSearchTask prepareLeaf(LeafReaderContext ctx, Weight filterWeight) throws IOException {
        ESAcceptDocs acceptDocs = buildLeafAcceptDocs(ctx, filterWeight);
        if (acceptDocs == null) {
            return null;
        }
        // Warm the lazy caches (filter bitset, cardinality, slice range) on this single preparing thread. Phase B
        // workers only read them; the executor join between Phase A and Phase B provides the happens-before edge.
        warmAcceptDocs(acceptDocs);

        SegmentReader segmentReader = Lucene.tryUnwrapSegmentReader(ctx.reader());
        if (segmentReader != null) {
            KnnVectorsReader vectorsReader = segmentReader.getVectorReader();
            if (vectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
                KnnVectorsReader fieldReader = fieldsReader.getFieldReader(field);
                if (fieldReader instanceof IVFVectorsReader<?> ivfReader) {
                    FieldInfo fieldInfo = segmentReader.getFieldInfos().fieldInfo(field);
                    if (fieldInfo != null) {
                        return prepareIvfLeaf(ivfReader, ctx, fieldInfo, acceptDocs);
                    }
                }
            }
        }
        // Not an IVF field (e.g. an HNSW segment in a mixed index): score through the regular per-leaf path.
        return LeafSearchTask.fallback(ctx, acceptDocs, providedVisitRatio);
    }

    private static void warmAcceptDocs(ESAcceptDocs acceptDocs) throws IOException {
        acceptDocs.bits();
        acceptDocs.cost();
        acceptDocs.approximateCost();
        if (acceptDocs.sliceOrd() >= 0) {
            acceptDocs.sliceAcceptDocs();
        }
    }

    /**
     * Selects the closest posting lists for an IVF leaf up to a per-leaf vector budget that mirrors the serial
     * codec path's two-loop selection ({@code maxVectorVisited} padded with {@code requiredToFill} so filtered
     * leaves still gather enough candidates). The generic parameter is captured from the {@code IVFVectorsReader<?>}
     * call site so the {@code E}-typed reader hooks ({@code getNumberOfVectors}, {@code estimatePostingVectorCount})
     * type-check. Returns {@code null} when the leaf has no vectors or selects no postings.
     */
    private <E extends IVFVectorsReader.FieldEntry> LeafSearchTask prepareIvfLeaf(
        IVFVectorsReader<E> reader,
        LeafReaderContext ctx,
        FieldInfo fieldInfo,
        ESAcceptDocs acceptDocs
    ) throws IOException {
        E entry = reader.fieldEntry(fieldInfo.number);
        if (entry == null) {
            return LeafSearchTask.fallback(ctx, acceptDocs, providedVisitRatio);
        }
        if (fieldInfo.getVectorDimension() != getQueryVector().length) {
            throw new IllegalArgumentException(
                "vector query dimension: " + getQueryVector().length + " differs from field dimension: " + fieldInfo.getVectorDimension()
            );
        }
        FloatVectorValues values = reader.getFloatVectorValues(field);
        IndexInput centroids = entry.centroidSlice(reader.ivfCentroids());
        int numVectors = reader.getNumberOfVectors(entry, values, centroids, acceptDocs);
        if (numVectors == 0) {
            return null;
        }
        final float approximateCost;
        if (acceptDocs instanceof ESAcceptDocs.ESAcceptDocsAll) {
            approximateCost = numVectors;
        } else {
            approximateCost = acceptDocs.approximateCost();
        }
        float percentFiltered = Math.max(0f, Math.min(1f, approximateCost / numVectors));
        float eff = reader.effectiveVisitRatio(providedVisitRatio, numCands, strategyK(), numVectors);

        // Account for soar vectors (a vector may be visited twice) with the 2x multiplier, exactly as the serial path.
        long maxVectorVisited = Math.max(1L, (long) Math.ceil(2.0 * eff * numVectors));
        if (percentFiltered > 0f && percentFiltered < 0.5f) {
            double scale = 0.5 / Math.max(percentFiltered, 0.25);
            maxVectorVisited = Math.min((long) numVectors, (long) Math.ceil(maxVectorVisited * scale));
        }
        int collectorK = Math.round(2f * k);
        long requiredToFill = percentFiltered > 0f ? (long) Math.ceil(collectorK / (double) percentFiltered) : numVectors;
        long targetVectors = Math.min((long) numVectors, Math.max(maxVectorVisited, requiredToFill));
        targetVectors = Math.max(1L, targetVectors);
        // When the budget reaches the whole leaf (typically a restrictive filter), select every posting. This mirrors
        // the serial path's secondary loop, which keeps scanning postings until it has scored k filtered docs; relying
        // on the (slightly over-counting) per-posting estimate to reach numVectors could otherwise stop a posting short.
        boolean scanAllPostings = targetVectors >= numVectors;

        IndexInput postListSlice = entry.postingListSlice(reader.ivfClusters());
        CentroidIterator it = reader.getCentroidIterator(
            fieldInfo,
            entry.numCentroids(),
            centroids,
            getQueryVector(),
            postListSlice,
            acceptDocs,
            approximateCost,
            values,
            eff
        );
        List<PostingSearchTask> postings = new ArrayList<>();
        long expectedDocs = 0;
        while (it.hasNext() && (scanAllPostings || expectedDocs < targetVectors)) {
            PostingMetadata md = it.nextPosting();
            int est = reader.estimatePostingVectorCount(entry, fieldInfo, md);
            postings.add(new PostingSearchTask(md, est));
            expectedDocs += est;
        }
        if (postings.isEmpty()) {
            return null;
        }
        return LeafSearchTask.ivf(
            ctx,
            acceptDocs,
            acceptDocs.bits(),
            reader,
            entry,
            fieldInfo,
            postings.toArray(new PostingSearchTask[0]),
            eff
        );
    }

    /**
     * Fused single-leaf scoring (the many-segments strategy): prepare the leaf (centroid scoring + posting
     * selection) and immediately score all of its selected postings on this one worker thread, with no inter-phase
     * barrier. Because the leaf is touched by exactly one thread, a plain clone of the warm shared mapping is used
     * (no thread-confined remap needed). The collector is still wired to the shared competitive frontier, so a leaf
     * that finishes early raises the global bar for the leaves still scoring. Returns an empty list when the leaf
     * has nothing to contribute, or a single {@link LeafResult}; falls back to the per-leaf approximate path for a
     * non-IVF leaf.
     */
    private List<LeafResult> scoreLeafFused(LeafReaderContext ctx, Weight filterWeight, IVFCollectorManager mgr) throws IOException {
        LeafSearchTask leaf = prepareLeaf(ctx, filterWeight);
        if (leaf == null) {
            return List.of();
        }
        if (leaf.needsFallbackSearch()) {
            return List.of(runFallback(leaf, mgr));
        }
        // A single worker scores this leaf, so it is never contended: a plain clone avoids the per-open cost of a
        // confined remap while still being free of cross-thread session contention.
        LeafWorkerState st = buildLeafWorkerState(leaf, false, mgr);
        try {
            if (st.collector() == null) {
                return List.of();
            }
            PostingSearchTask[] postings = leaf.postings();
            // Two passes: prefetch every posting first, then score, so I/O overlaps with scoring.
            for (PostingSearchTask posting : postings) {
                PostingMetadata md = posting.metadata();
                st.postListSlice().prefetch(md.offset(), md.length());
            }
            for (PostingSearchTask posting : postings) {
                PostingMetadata md = posting.metadata();
                st.visitor().resetPostingsScorer(md);
                st.visitor().visit(st.collector());
                st.strategy().nextVectorsBlock();
            }
            TopDocs topDocs = st.collector() instanceof BulkKnnCollector bulk ? bulk.unsortedTopK() : st.collector().topDocs();
            if (topDocs != null && topDocs.scoreDocs.length > 0) {
                return List.of(new LeafResult(st.docBase(), topDocs));
            }
            return List.of();
        } finally {
            IOUtils.close(st);
        }
    }

    /**
     * Phase B worker: drains the shared lock-free {@code cursor} over the pre-sorted {@code slices}, scoring each
     * slice through per-leaf state that is created lazily and confined to this worker thread. Two passes per slice:
     * first prefetch every posting, then score them so I/O overlaps. All thread-confined handles are closed in the
     * {@code finally} on this same worker thread, as the confined-arena contract requires.
     */
    private List<LeafResult> runWorker(
        Slice[] slices,
        AtomicInteger cursor,
        boolean[] contended,
        List<LeafSearchTask> ivfLeaves,
        IVFCollectorManager mgr
    ) throws IOException {
        Map<Integer, LeafWorkerState> states = new HashMap<>();
        try {
            int idx;
            while ((idx = cursor.getAndIncrement()) < slices.length) {
                Slice slice = slices[idx];
                // Prefetch the current slice (building per-leaf state as needed).
                prefetchSlice(slice, states, ivfLeaves, contended, mgr, true);
                // Speculatively prefetch the next slice so its I/O overlaps with scoring below, but only for leaves
                // whose state this worker already holds (do not build state for a slice another worker will claim —
                // with N workers the peeked slice is usually not ours). This helps the single-leaf case (consecutive
                // same-leaf bands) without wasting handle/visitor setup in the multi-leaf case.
                int nextPeek = cursor.get();
                if (nextPeek < slices.length) {
                    prefetchSlice(slices[nextPeek], states, ivfLeaves, contended, mgr, false);
                }
                // Score every posting in the slice, tightening the shared frontier after each.
                for (SliceEntry entry : slice.entries()) {
                    LeafWorkerState st = states.get(entry.leafIndex());
                    if (st == null || st.collector() == null) {
                        continue;
                    }
                    PostingSearchTask[] postings = ivfLeaves.get(entry.leafIndex()).postings();
                    for (int p = entry.postingStart(); p < entry.postingEnd(); p++) {
                        PostingMetadata md = postings[p].metadata();
                        st.visitor().resetPostingsScorer(md);
                        st.visitor().visit(st.collector());
                        st.strategy().nextVectorsBlock();
                    }
                }
            }
            List<LeafResult> results = new ArrayList<>();
            for (LeafWorkerState st : states.values()) {
                if (st.collector() == null) {
                    continue;
                }
                TopDocs topDocs = st.collector() instanceof BulkKnnCollector bulk ? bulk.unsortedTopK() : st.collector().topDocs();
                if (topDocs != null && topDocs.scoreDocs.length > 0) {
                    results.add(new LeafResult(st.docBase(), topDocs));
                }
            }
            return results;
        } finally {
            IOUtils.close(states.values());
        }
    }

    private void prefetchSlice(
        Slice slice,
        Map<Integer, LeafWorkerState> states,
        List<LeafSearchTask> ivfLeaves,
        boolean[] contended,
        IVFCollectorManager mgr,
        boolean buildIfMissing
    ) throws IOException {
        for (SliceEntry entry : slice.entries()) {
            LeafWorkerState st = buildIfMissing
                ? ensureState(states, entry.leafIndex(), ivfLeaves, contended, mgr)
                : states.get(entry.leafIndex());
            if (st == null || st.collector() == null) {
                continue;
            }
            PostingSearchTask[] postings = ivfLeaves.get(entry.leafIndex()).postings();
            for (int p = entry.postingStart(); p < entry.postingEnd(); p++) {
                PostingMetadata md = postings[p].metadata();
                st.postListSlice().prefetch(md.offset(), md.length());
            }
        }
    }

    private LeafWorkerState ensureState(
        Map<Integer, LeafWorkerState> states,
        int leafIndex,
        List<LeafSearchTask> ivfLeaves,
        boolean[] contended,
        IVFCollectorManager mgr
    ) throws IOException {
        LeafWorkerState st = states.get(leafIndex);
        if (st == null) {
            st = buildLeafWorkerState(ivfLeaves.get(leafIndex), contended[leafIndex], mgr);
            states.put(leafIndex, st);
        }
        return st;
    }

    /**
     * Builds this worker's private scoring state for one leaf. A contended leaf (scored by more than one worker)
     * gets thread-confined mappings for both clusters and centroids to avoid cross-thread CAS on shared
     * {@code MemorySegment} sessions; a non-contended leaf simply clones the warm shared mappings. When the
     * collector manager yields no collector, an empty, handle-free state is returned.
     */
    private LeafWorkerState buildLeafWorkerState(LeafSearchTask leaf, boolean contended, IVFCollectorManager mgr) throws IOException {
        IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(leaf.visitRatio(), numCands, strategyK(), mgr.longAccumulator);
        AbstractMaxScoreKnnCollector collector = mgr.newCollector(Integer.MAX_VALUE, strategy, leaf.ctx());
        if (collector == null) {
            return LeafWorkerState.empty(leaf.ctx().docBase);
        }
        strategy.setCollector(collector);
        IVFVectorsReader<?> reader = leaf.reader();
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
            IndexInput postListSlice = leaf.entry().postingListSlice(clustersHandle);
            IndexInput centroidSlice = leaf.entry().centroidSlice(centroidsHandle);
            FloatVectorValues values = reader.getFloatVectorValues(field);
            IVFVectorsReader.PostingVisitor visitor = reader.getPostingVisitor(
                leaf.fieldInfo(),
                values,
                postListSlice,
                getQueryVector(),
                leaf.materializedFilter(),
                centroidSlice,
                leaf.acceptDocs()
            );
            return new LeafWorkerState(clustersHandle, centroidsHandle, postListSlice, visitor, collector, strategy, leaf.ctx().docBase);
        } catch (Throwable t) {
            IOUtils.closeWhileHandlingException(clustersHandle, centroidsHandle);
            throw t;
        }
    }

    /** Scores a non-IVF (fallback) leaf through the regular per-leaf approximate-search path. */
    private LeafResult runFallback(LeafSearchTask leaf, IVFCollectorManager mgr) throws IOException {
        TopDocs topDocs = approximateSearch(leaf.ctx(), leaf.acceptDocs(), Integer.MAX_VALUE, mgr, leaf.visitRatio());
        return new LeafResult(leaf.ctx().docBase, topDocs);
    }

    /**
     * Phase C: merge every worker's per-leaf results into the global top {@code mergeK}. Results are deduplicated
     * by absolute document id keeping the highest score — a document can recur across overspill postings, and a
     * single leaf's postings may be split across several workers, so the same document can be collected more than
     * once. Scoring is deterministic, so the kept score is identical regardless of which worker produced it.
     */
    private TopDocs mergeResults(int mergeK, List<LeafResult> leafResults) {
        IntObjectHashMap<ScoreDoc> dedupByDoc = new IntObjectHashMap<>();
        long totalHitsValue = 0;
        TotalHits.Relation relation = TotalHits.Relation.EQUAL_TO;
        for (LeafResult leafResult : leafResults) {
            TopDocs topDocs = leafResult.topDocs();
            totalHitsValue += topDocs.totalHits.value();
            if (topDocs.totalHits.relation() == TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO) {
                relation = TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO;
            }
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                int globalDoc = scoreDoc.doc + leafResult.docBase();
                ScoreDoc existing = dedupByDoc.get(globalDoc);
                if (existing == null) {
                    scoreDoc.doc = globalDoc;
                    dedupByDoc.put(globalDoc, scoreDoc);
                } else if (scoreDoc.score > existing.score) {
                    existing.score = scoreDoc.score;
                }
            }
        }
        if (dedupByDoc.isEmpty()) {
            return new TopDocs(new TotalHits(totalHitsValue, relation), new ScoreDoc[0]);
        }
        int count = dedupByDoc.size();
        int[] docs = new int[count];
        float[] scores = new float[count];
        float bestScore = Float.NEGATIVE_INFINITY;
        int i = 0;
        for (IntObjectHashMap.IntObjectCursor<ScoreDoc> cursor : dedupByDoc) {
            docs[i] = cursor.value.doc;
            scores[i] = cursor.value.score;
            if (cursor.value.score > bestScore) {
                bestScore = cursor.value.score;
            }
            i++;
        }
        BulkNeighborQueue mergeQueue = BulkNeighborQueue.forMerging(mergeK);
        mergeQueue.insertWithOverflowBulk(docs, scores, count, bestScore);
        ScoreDoc[] mergedScoreDocs = new ScoreDoc[mergeQueue.size()];
        int[] index = new int[] { mergedScoreDocs.length - 1 };
        mergeQueue.drain(
            encoded -> mergedScoreDocs[index[0]--] = new ScoreDoc(mergeQueue.decodeNodeId(encoded), mergeQueue.decodeScore(encoded))
        );
        return new TopDocs(new TotalHits(totalHitsValue, relation), mergedScoreDocs);
    }

    /**
     * Resolves the {@link ESAcceptDocs} for a leaf from the (optional) filter weight. Returns {@code null} to skip a
     * leaf that cannot contribute (the filter matches nothing here). Overridden by the sliced query to additionally
     * restrict the doc-id space to a single slice.
     */
    protected ESAcceptDocs buildLeafAcceptDocs(LeafReaderContext ctx, Weight filterWeight) throws IOException {
        final LeafReader reader = ctx.reader();
        final Bits liveDocs = reader.getLiveDocs();
        final int maxDoc = reader.maxDoc();
        if (filterWeight == null) {
            return liveDocs == null ? new ESAcceptDocs.ESAcceptDocsAll() : new ESAcceptDocs.BitsAcceptDocs(liveDocs, maxDoc);
        }
        ScorerSupplier supplier = filterWeight.scorerSupplier(ctx);
        if (supplier == null) {
            return null;
        }
        return new ESAcceptDocs.ScorerSupplierAcceptDocs(supplier, liveDocs, maxDoc);
    }

    /**
     * The k used to size the search strategy and the codec's dynamic visit ratio — the pre-oversampling k. Defaults
     * to {@link #k}; the float query overrides it with the original (un-oversampled) k.
     */
    protected int strategyK() {
        return k;
    }

    /** The (possibly preconditioned) query vector to score against. */
    protected abstract float[] getQueryVector();

    /**
     * When {@code true}, Phase B emits exactly one slice per leaf (no banding or merging) so each leaf is scored by a
     * single worker. Required by nested diversification, where splitting a leaf across workers would let two children
     * of the same parent survive in different collectors.
     */
    protected boolean requiresWholeLeafSlices() {
        return false;
    }

    abstract void preconditionQuery(LeafReaderContext context) throws IOException;

    abstract TopDocs approximateSearch(
        LeafReaderContext context,
        AcceptDocs acceptDocs,
        int visitedLimit,
        IVFCollectorManager knnCollectorManager,
        float visitRatio
    ) throws IOException;

    protected IVFCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
        return new IVFCollectorManager(k, searcher);
    }

    @Override
    public final void profile(QueryProfiler queryProfiler) {
        queryProfiler.addVectorOpsCount(vectorOpsCount);
    }

    /** A single worker's scored results for one leaf, carried to Phase C with the leaf's {@code docBase}. */
    private record LeafResult(int docBase, TopDocs topDocs) {}

    /**
     * One worker's private, thread-confined scoring state for a single leaf. Created, used, and closed entirely on
     * the worker thread. {@code collector} is {@code null} (and no handles are opened) when the leaf yields no
     * collector. Closing releases the posting-list and centroid handles; the slices derived from them need no
     * separate close.
     */
    private record LeafWorkerState(
        IndexInput clustersHandle,
        IndexInput centroidsClone,
        IndexInput postListSlice,
        IVFVectorsReader.PostingVisitor visitor,
        AbstractMaxScoreKnnCollector collector,
        IVFKnnSearchStrategy strategy,
        int docBase
    ) implements Closeable {
        static LeafWorkerState empty(int docBase) {
            return new LeafWorkerState(null, null, null, null, null, null, docBase);
        }

        @Override
        public void close() throws IOException {
            IOUtils.close(clustersHandle, centroidsClone);
        }
    }

    static class IVFCollectorManager implements KnnCollectorManager {
        private final int k;
        final LongAccumulator longAccumulator;

        IVFCollectorManager(int k, IndexSearcher searcher) {
            this.k = k;
            // The competitive frontier is shared across every (worker, leaf) strategy so that the posting-granular
            // workers scoring a single segment in parallel prune against the global best, not just their local view.
            longAccumulator = new LongAccumulator(Long::max, LEAST_COMPETITIVE);
        }

        @Override
        public AbstractMaxScoreKnnCollector newCollector(int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context)
            throws IOException {
            return new MaxScoreTopKnnCollector(k, visitedLimit, searchStrategy);
        }
    }
}
