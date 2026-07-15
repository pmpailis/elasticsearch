/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
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
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.core.Booleans;
import org.elasticsearch.index.codec.vectors.cluster.BulkNeighborQueue;
import org.elasticsearch.index.codec.vectors.diskbbq.CrossSegmentPostingScheduler;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfQueryConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfSegmentConfig;
import org.elasticsearch.search.internal.ContextIndexSearcher;
import org.elasticsearch.search.profile.query.QueryProfiler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.LongAccumulator;

import static org.elasticsearch.search.vectors.AbstractMaxScoreKnnCollector.LEAST_COMPETITIVE;

/**
 * Base class for IVF kNN vector queries. {@link #k} is the final result size (after any outer rescore); per-segment
 * preconditioning and oversample expansion come from {@link IvfQueryConfigResolver#resolve}.
 */
abstract class AbstractIVFKnnVectorQuery extends Query implements QueryProfilerProvider {

    static final TopDocs NO_RESULTS = TopDocsCollector.EMPTY_TOPDOCS;

    /**
     * Within-segment parallel scanning of IVF posting lists. On by default; the sysprop is a runtime kill switch
     * (set to {@code false} to force the serial per-segment scan).
     */
    static final boolean INTRA_SEGMENT_PARALLELISM_ENABLED = Booleans.parseBoolean(
        System.getProperty("es.vectors.ivf_intra_segment_parallelism"),
        true
    );

    /** Upper bound on parallel workers per segment; the effective count also divides by the number of leaves. */
    static final int MAX_INTRA_SEGMENT_WORKERS = Integer.parseInt(System.getProperty("es.vectors.ivf_max_intra_segment_workers", "8"));

    /**
     * Benchmarking/dev escape hatch: forces this many workers per segment when the searcher is not a
     * {@link ContextIndexSearcher} (e.g. the qa/vector {@code KnnSearcher} harness uses a plain executor-backed
     * {@code IndexSearcher}). {@code 0} (the default) disables the fallback; production searches always go through
     * {@link ContextIndexSearcher} and are unaffected.
     */
    static final int FORCE_INTRA_SEGMENT_WORKERS = Integer.parseInt(System.getProperty("es.vectors.ivf_force_intra_segment_workers", "0"));

    /**
     * Sharing of the min-competitive score floor across segments, so a segment holding no globally competitive
     * documents stops collecting as soon as other segments have produced better results. Independent of
     * {@link #INTRA_SEGMENT_PARALLELISM_ENABLED}: it also benefits the serial multi-segment path. On by default;
     * the sysprop is a runtime kill switch. Lossless by construction (floors are published under the SOAR-safe
     * distinct-doc gate and only prune docs that cannot make the merged top-k).
     */
    static final boolean CROSS_LEAF_FLOOR_SHARING_ENABLED = Booleans.parseBoolean(
        System.getProperty("es.vectors.ivf_cross_leaf_floor_sharing"),
        true
    );

    /**
     * Cross-segment best-first scheduling: instead of scanning each segment independently under its own visit
     * budget, eligible segments register their ranked postings with a query-level scheduler whose workers claim the
     * globally most promising chunk next, spending the summed budget where the centroid scores say it matters.
     * Requires {@link #INTRA_SEGMENT_PARALLELISM_ENABLED}. Off by default: on uniformly distributed corpora the
     * per-leaf task parallelism of the baseline already saturates the pool with zero coordination, and the
     * scheduler's phase barriers and per-(worker, leaf) scoring state measured 5-25% slower at equal recall.
     * Opt in for segment layouts with real cross-segment query affinity (e.g. time-based or routed indices).
     */
    static final boolean CROSS_SEGMENT_SCHEDULING_ENABLED = Booleans.parseBoolean(
        System.getProperty("es.vectors.ivf_cross_segment_scheduling"),
        false
    );

    protected final String field;
    protected final float providedVisitRatio;
    protected final int k;
    protected final int numCands;
    protected final Query filter;
    protected int vectorOpsCount;
    protected final IvfQueryConfigResolver ivfQueryConfigResolver;
    private IVFParallelismConfig intraSegmentParallelism;
    private CrossSegmentPostingScheduler crossSegmentScheduler;

    protected AbstractIVFKnnVectorQuery(
        String field,
        float visitRatio,
        int k,
        int numCands,
        Query filter,
        IvfQueryConfigResolver ivfQueryConfigResolver
    ) {
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
        this.ivfQueryConfigResolver = Objects.requireNonNull(ivfQueryConfigResolver, "ivfQueryConfigResolver should not be null");
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

        TaskExecutor taskExecutor = indexSearcher.getTaskExecutor();
        List<LeafReaderContext> leafReaderContexts = reader.leaves();
        intraSegmentParallelism = resolveIntraSegmentParallelism(indexSearcher, leafReaderContexts);

        // When providedVisitRatio is 0.0f (dynamic), the codec computes the visit ratio
        // per-segment using the Two-Signal model with segment-size awareness.
        final float visitRatio = providedVisitRatio;

        // Pass 1: resolve per-leaf configs first, so the shard merge budget (mergeK) — which gates cross-leaf floor
        // publishing — is known before any collector manager is created.
        IvfSegmentConfig[] resolvedBySegment = new IvfSegmentConfig[leafReaderContexts.size()];
        float maxRescoreOversampleAcrossLeaves = 1f;
        int leavesWithField = 0;
        for (int i = 0; i < leafReaderContexts.size(); i++) {
            LeafReader leafReader = leafReaderContexts.get(i).reader();
            FieldInfo fieldInfo = leafReader.getFieldInfos().fieldInfo(field);
            if (fieldInfo != null) {
                leavesWithField++;
                IvfSegmentConfig resolved = ivfQueryConfigResolver.resolve(fieldInfo, leafReader);
                resolvedBySegment[i] = resolved;
                maxRescoreOversampleAcrossLeaves = Math.max(maxRescoreOversampleAcrossLeaves, resolved.rescoreOversample());
            }
        }
        int mergeK = leafReaderContexts.isEmpty() ? k : IvfSegmentConfig.shardMergeBudget(k, maxRescoreOversampleAcrossLeaves);
        // One query-level floor shared by every leaf's collectors, so segments prune against each other's results.
        LongAccumulator globalFloor = crossLeafFloorSharing() && leavesWithField > 1
            ? new LongAccumulator(Long::max, LEAST_COMPETITIVE)
            : null;
        // Cross-segment best-first scheduling: eligible leaves register their ranked postings instead of scanning;
        // the scheduler spends the summed visit budget on the globally most promising chunks after the leaf tasks
        // complete. The worker budget is NOT divided by leaf count here — scheduler workers replace leaf-level
        // scanning rather than running alongside it.
        IVFParallelismConfig parallelismConfig = intraSegmentParallelism();
        crossSegmentScheduler = crossSegmentScheduling() && parallelismConfig != null && leavesWithField > 1
            ? new CrossSegmentPostingScheduler(
                parallelismConfig.taskExecutor(),
                parallelismConfig.schedulerWorkers(),
                parallelismConfig.checkCancelled()
            )
            : null;

        // Pass 2: build one task per leaf.
        List<Callable<TopDocs>> tasks = new ArrayList<>(leafReaderContexts.size());
        for (int i = 0; i < leafReaderContexts.size(); i++) {
            final LeafReaderContext context = leafReaderContexts.get(i);
            IvfSegmentConfig resolved = resolvedBySegment[i];
            float segmentOversample = resolved != null ? resolved.rescoreOversample() : 1f;

            IVFCollectorManager knnCollectorManagerForSegment = getKnnCollectorManager(
                IvfSegmentConfig.leafCollectorBudget(k, segmentOversample),
                indexSearcher,
                globalFloor,
                mergeK
            );

            // Preconditioning is resolved per leaf inside the task: a shard may mix preconditioned and
            // non-preconditioned segments, and each must be searched with the query view matching its own space.
            final boolean usePrecondition = resolved != null && resolved.usePrecondition();
            tasks.add(() -> searchLeaf(context, filterWeight, knnCollectorManagerForSegment, visitRatio, usePrecondition));
        }
        TopDocs[] perLeafResults = taskExecutor.invokeAll(tasks).toArray(TopDocs[]::new);

        if (crossSegmentScheduler != null && crossSegmentScheduler.run()) {
            // Registered leaves returned a null marker from their task; their results exist only now.
            for (int i = 0; i < perLeafResults.length; i++) {
                if (perLeafResults[i] == null) {
                    perLeafResults[i] = dedupLeafResults(crossSegmentScheduler.topDocs(i), leafReaderContexts.get(i).docBase);
                }
            }
        }

        TopDocs topK = mergeLeafResults(mergeK, perLeafResults);
        vectorOpsCount = (int) topK.totalHits.value();
        if (topK.scoreDocs.length == 0) {
            return Queries.NO_DOCS_INSTANCE;
        }
        if (ivfQueryConfigResolver.isAutoCalibrate()) {
            return getAutoRescoreQuery(indexSearcher, topK, mergeK);
        }
        return new KnnScoreDocQuery(topK.scoreDocs, reader);
    }

    /**
     * Returns a query that performs exact rescoring of oversampled candidates.
     * Implementations can return {@code null} when rescoring is unavailable.
     */
    abstract Query getAutoRescoreQuery(IndexSearcher indexSearcher, TopDocs topOversampled, int effectiveK);

    private TopDocs mergeLeafResults(int mergeK, TopDocs[] perLeafResults) {
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

    private TopDocs searchLeaf(
        LeafReaderContext ctx,
        Weight filterWeight,
        IVFCollectorManager knnCollectorManager,
        float visitRatio,
        boolean usePrecondition
    ) throws IOException {
        TopDocs results = getLeafResults(ctx, filterWeight, knnCollectorManager, visitRatio, usePrecondition);
        if (results == null) {
            // the leaf registered with the cross-segment scheduler; its results are collected after it runs
            return null;
        }
        return dedupLeafResults(results, ctx.docBase);
    }

    /** Maps doc ids to global space, keeping the best-scoring copy of each SOAR-duplicated doc. */
    private static TopDocs dedupLeafResults(TopDocs results, int docBase) {
        IntObjectHashMap<ScoreDoc> dedupByDoc = new IntObjectHashMap<>(results.scoreDocs.length * 4 / 3);
        for (ScoreDoc scoreDoc : results.scoreDocs) {
            int globalDoc = scoreDoc.doc + docBase;
            // Keep the best-scoring copy of a SOAR-duplicated doc (each posting quantizes against its own centroid,
            // so the copies score differently). Max-wins is insertion-order independent, which keeps results
            // deterministic when intra-segment parallel scanning changes the queue's drain order.
            ScoreDoc existing = dedupByDoc.get(globalDoc);
            if (existing == null || scoreDoc.score > existing.score) {
                scoreDoc.doc = globalDoc;
                dedupByDoc.put(globalDoc, scoreDoc);
            }
        }
        ScoreDoc[] deduplicatedScoreDocs = new ScoreDoc[dedupByDoc.size()];
        int index = 0;
        for (IntObjectHashMap.IntObjectCursor<ScoreDoc> deduplicated : dedupByDoc) {
            deduplicatedScoreDocs[index++] = deduplicated.value;
        }
        return new TopDocs(results.totalHits, deduplicatedScoreDocs);
    }

    abstract TopDocs getLeafResults(
        LeafReaderContext ctx,
        Weight filterWeight,
        IVFCollectorManager knnCollectorManager,
        float visitRatio,
        boolean usePrecondition
    ) throws IOException;

    protected IVFCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher, LongAccumulator globalFloor, int mergeK) {
        return new IVFCollectorManager(k, searcher, globalFloor, mergeK);
    }

    /** The per-query parallelism resolved by the last {@link #rewrite}, or {@code null} when disabled. */
    IVFParallelismConfig intraSegmentParallelism() {
        return intraSegmentParallelism;
    }

    /** Overridable for tests: the flag is a static final read at class load, unreliable in a shared test JVM. */
    boolean crossLeafFloorSharing() {
        return CROSS_LEAF_FLOOR_SHARING_ENABLED;
    }

    /** Overridable for tests, like {@link #crossLeafFloorSharing()}. */
    boolean crossSegmentScheduling() {
        return CROSS_SEGMENT_SCHEDULING_ENABLED;
    }

    /** The query-level scheduler created by the last {@link #rewrite}, or {@code null} when disabled. */
    final CrossSegmentPostingScheduler crossSegmentScheduler() {
        return crossSegmentScheduler;
    }

    /**
     * Per-leaf context for within-segment parallel scanning of posting lists, or {@code null} when intra-segment
     * parallelism is disabled for this query. Workers share the manager's {@link ScoreFloors} when one exists —
     * coupling them with the leaf collector and, when cross-leaf sharing is enabled, with the other leaves —
     * otherwise a leaf-local instance, so workers always prune against each other's results.
     */
    final IVFParallelScanContext newParallelScanContext(IVFCollectorManager knnCollectorManager) {
        IVFParallelismConfig parallelismConfig = intraSegmentParallelism();
        if (parallelismConfig == null || parallelismConfig.maxWorkers() < 2) {
            // the config may exist solely for the cross-segment scheduler; per-leaf forking needs >= 2 workers
            return null;
        }
        ScoreFloors floors = knnCollectorManager.floors != null ? knnCollectorManager.floors : new ScoreFloors(null, false);
        return new IVFParallelScanContext(
            parallelismConfig.taskExecutor(),
            parallelismConfig.maxWorkers(),
            parallelismConfig.checkCancelled(),
            floors
        );
    }

    private IVFParallelismConfig resolveIntraSegmentParallelism(IndexSearcher searcher, List<LeafReaderContext> leaves) {
        if (INTRA_SEGMENT_PARALLELISM_ENABLED == false) {
            return null;
        }
        int leavesWithField = 0;
        for (LeafReaderContext leafContext : leaves) {
            if (leafContext.reader().getFieldInfos().fieldInfo(field) != null) {
                leavesWithField++;
            }
        }
        if (leavesWithField == 0) {
            return null;
        }
        if (searcher instanceof ContextIndexSearcher contextIndexSearcher && contextIndexSearcher.hasExecutor()) {
            // Divide the request-level concurrency budget across leaves so leaf tasks and intra-segment workers
            // together do not oversubscribe the search pool; multi-segment indices naturally degrade to one worker.
            // Cross-segment scheduler workers are NOT divided by leaf count (they replace leaf-level scanning).
            int concurrency = Math.max(1, contextIndexSearcher.getMaximumNumberOfSlices());
            int maxWorkers = Math.min(MAX_INTRA_SEGMENT_WORKERS, Math.max(1, concurrency / leavesWithField));
            int schedulerWorkers = Math.min(MAX_INTRA_SEGMENT_WORKERS, concurrency);
            if (maxWorkers >= 2 || (schedulerWorkers >= 2 && crossSegmentScheduling())) {
                return new IVFParallelismConfig(
                    searcher.getTaskExecutor(),
                    maxWorkers,
                    schedulerWorkers,
                    contextIndexSearcher::checkCancelled
                );
            }
        } else if (FORCE_INTRA_SEGMENT_WORKERS >= 2) {
            // Plain IndexSearcher (Lucene-level harnesses): no cancellation hook exists, and Lucene's TaskExecutor
            // degrades to caller-runs when the searcher has no executor, so this stays correct either way.
            return new IVFParallelismConfig(searcher.getTaskExecutor(), FORCE_INTRA_SEGMENT_WORKERS, FORCE_INTRA_SEGMENT_WORKERS, () -> {});
        }
        return null;
    }

    record IVFParallelismConfig(TaskExecutor taskExecutor, int maxWorkers, int schedulerWorkers, Runnable checkCancelled) {

        IVFParallelismConfig(TaskExecutor taskExecutor, int maxWorkers, Runnable checkCancelled) {
            this(taskExecutor, maxWorkers, maxWorkers, checkCancelled);
        }
    }

    @Override
    public final void profile(QueryProfiler queryProfiler) {
        queryProfiler.addVectorOpsCount(vectorOpsCount);
    }

    static class IVFCollectorManager implements KnnCollectorManager {
        private final int k;
        final ScoreFloors floors;

        IVFCollectorManager(int k, IndexSearcher searcher, LongAccumulator globalFloor, int mergeK) {
            // A plain top-k heap holds up to two SOAR entries per doc, so k entries certify only ceil(k/2) distinct
            // docs above the floor: only leaves whose collector budget covers 2*mergeK distinct docs may export
            // their floor to other leaves. (Diversifying heaps hold distinct parents and gate at mergeK instead.)
            this(k, searcher, globalFloor, k >= 2 * mergeK);
        }

        IVFCollectorManager(int k, IndexSearcher searcher, LongAccumulator globalFloor, boolean publishGlobalFloor) {
            this.k = k;
            // Floors exist whenever collectors may exchange them: with sibling leaves, or an explicit global floor.
            // Intra-segment workers on a single-leaf index get a leaf-local fallback in newParallelScanContext.
            this.floors = searcher.getIndexReader().leaves().size() > 1 || globalFloor != null
                ? new ScoreFloors(globalFloor, publishGlobalFloor)
                : null;
        }

        @Override
        public AbstractMaxScoreKnnCollector newCollector(int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context)
            throws IOException {
            return new MaxScoreTopKnnCollector(k, visitedLimit, searchStrategy);
        }
    }
}
