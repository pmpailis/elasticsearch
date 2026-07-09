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
     * Experimental opt-in for within-segment parallel scanning of IVF posting lists. Strictly off by default (this is
     * deliberately not a {@link org.elasticsearch.common.util.FeatureFlag}, which would auto-enable on snapshot
     * builds) until benchmarks validate the parallel path.
     */
    static final boolean INTRA_SEGMENT_PARALLELISM_ENABLED = Booleans.parseBoolean(
        System.getProperty("es.vectors.ivf_intra_segment_parallelism"),
        false
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

    protected final String field;
    protected final float providedVisitRatio;
    protected final int k;
    protected final int numCands;
    protected final Query filter;
    protected int vectorOpsCount;
    protected final IvfQueryConfigResolver ivfQueryConfigResolver;
    private IVFParallelismConfig intraSegmentParallelism;

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

        List<Callable<TopDocs>> tasks = new ArrayList<>(leafReaderContexts.size());
        float maxRescoreOversampleAcrossLeaves = 1f;
        for (LeafReaderContext context : leafReaderContexts) {
            LeafReader leafReader = context.reader();
            FieldInfo fieldInfo = leafReader.getFieldInfos().fieldInfo(field);
            float segmentOversample = 1f;
            IvfSegmentConfig resolved = null;
            if (fieldInfo != null) {
                resolved = ivfQueryConfigResolver.resolve(fieldInfo, leafReader);
                segmentOversample = resolved.rescoreOversample();
                maxRescoreOversampleAcrossLeaves = Math.max(maxRescoreOversampleAcrossLeaves, segmentOversample);
            }

            IVFCollectorManager knnCollectorManagerForSegment = getKnnCollectorManager(
                IvfSegmentConfig.leafCollectorBudget(k, segmentOversample),
                indexSearcher
            );

            if (resolved != null && resolved.usePrecondition()) {
                preconditionQuery(context);
            }
            tasks.add(() -> searchLeaf(context, filterWeight, knnCollectorManagerForSegment, visitRatio));
        }
        TopDocs[] perLeafResults = taskExecutor.invokeAll(tasks).toArray(TopDocs[]::new);

        int mergeK = tasks.isEmpty() ? k : IvfSegmentConfig.shardMergeBudget(k, maxRescoreOversampleAcrossLeaves);
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

    private TopDocs searchLeaf(LeafReaderContext ctx, Weight filterWeight, IVFCollectorManager knnCollectorManager, float visitRatio)
        throws IOException {
        TopDocs results = getLeafResults(ctx, filterWeight, knnCollectorManager, visitRatio);
        IntObjectHashMap<ScoreDoc> dedupByDoc = new IntObjectHashMap<>(results.scoreDocs.length * 4 / 3);
        for (ScoreDoc scoreDoc : results.scoreDocs) {
            int globalDoc = scoreDoc.doc + ctx.docBase;
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

    abstract TopDocs getLeafResults(LeafReaderContext ctx, Weight filterWeight, IVFCollectorManager knnCollectorManager, float visitRatio)
        throws IOException;

    abstract void preconditionQuery(LeafReaderContext context) throws IOException;

    protected IVFCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
        return new IVFCollectorManager(k, searcher);
    }

    /** The per-query parallelism resolved by the last {@link #rewrite}, or {@code null} when disabled. */
    IVFParallelismConfig intraSegmentParallelism() {
        return intraSegmentParallelism;
    }

    /**
     * Per-leaf context for within-segment parallel scanning of posting lists, or {@code null} when intra-segment
     * parallelism is disabled for this query. Workers share the manager's accumulator when one exists (note it is
     * per-leaf: managers are created per leaf in {@link #rewrite}), otherwise a fresh one, so workers always prune
     * against each other's results.
     */
    final IVFParallelScanContext newParallelScanContext(IVFCollectorManager knnCollectorManager) {
        IVFParallelismConfig parallelismConfig = intraSegmentParallelism();
        if (parallelismConfig == null) {
            return null;
        }
        LongAccumulator workerAccumulator = knnCollectorManager.longAccumulator != null
            ? knnCollectorManager.longAccumulator
            : new LongAccumulator(Long::max, LEAST_COMPETITIVE);
        return new IVFParallelScanContext(
            parallelismConfig.taskExecutor(),
            parallelismConfig.maxWorkers(),
            parallelismConfig.checkCancelled(),
            workerAccumulator
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
            int concurrency = Math.max(1, contextIndexSearcher.getMaximumNumberOfSlices());
            int maxWorkers = Math.min(MAX_INTRA_SEGMENT_WORKERS, Math.max(1, concurrency / leavesWithField));
            if (maxWorkers >= 2) {
                return new IVFParallelismConfig(searcher.getTaskExecutor(), maxWorkers, contextIndexSearcher::checkCancelled);
            }
        } else if (FORCE_INTRA_SEGMENT_WORKERS >= 2) {
            // Plain IndexSearcher (Lucene-level harnesses): no cancellation hook exists, and Lucene's TaskExecutor
            // degrades to caller-runs when the searcher has no executor, so this stays correct either way.
            return new IVFParallelismConfig(searcher.getTaskExecutor(), FORCE_INTRA_SEGMENT_WORKERS, () -> {});
        }
        return null;
    }

    record IVFParallelismConfig(TaskExecutor taskExecutor, int maxWorkers, Runnable checkCancelled) {}

    @Override
    public final void profile(QueryProfiler queryProfiler) {
        queryProfiler.addVectorOpsCount(vectorOpsCount);
    }

    static class IVFCollectorManager implements KnnCollectorManager {
        private final int k;
        final LongAccumulator longAccumulator;

        IVFCollectorManager(int k, IndexSearcher searcher) {
            this.k = k;
            longAccumulator = searcher.getIndexReader().leaves().size() > 1 ? new LongAccumulator(Long::max, LEAST_COMPETITIVE) : null;
        }

        @Override
        public AbstractMaxScoreKnnCollector newCollector(int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context)
            throws IOException {
            return new MaxScoreTopKnnCollector(k, visitedLimit, searchStrategy);
        }
    }
}
