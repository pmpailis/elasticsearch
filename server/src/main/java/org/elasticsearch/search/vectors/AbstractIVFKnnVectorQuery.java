/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
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
import org.apache.lucene.util.Bits;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.index.codec.vectors.cluster.BulkNeighborQueue;
import org.elasticsearch.index.codec.vectors.diskbbq.BatchedPostingVisitor;
import org.elasticsearch.index.codec.vectors.diskbbq.PostingMetadata;
import org.elasticsearch.search.profile.query.QueryProfiler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAccumulator;

import static org.elasticsearch.search.vectors.AbstractMaxScoreKnnCollector.LEAST_COMPETITIVE;

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

        // we request numCands as we are using it as an approximation measure
        // we need to ensure we are getting at least 2*k results to ensure we cover overspill duplicates
        // TODO move the logic for automatically adjusting percentages to the query, so we can only pass
        // 2k to the collector.
        IVFCollectorManager knnCollectorManager = getKnnCollectorManager(Math.round(2f * k), indexSearcher);
        TaskExecutor taskExecutor = indexSearcher.getTaskExecutor();
        List<LeafReaderContext> leafReaderContexts = reader.leaves();

        // When providedVisitRatio is 0.0f (dynamic), the codec computes the visit ratio
        // per-segment using the Two-Signal model with segment-size awareness.
        final float visitRatio = providedVisitRatio;

        // Phase 1: build a per-leaf BatchedPostingVisitor in parallel. Each holds the leaf's
        // CentroidIterator + PostingVisitor + posting-list slice and drives adaptive prefetch.
        // Leaves whose vector format is not IVF (e.g. an HNSW index queried through an IVF
        // query — exercised by getStableIndexStore tests) come back as null and fall back to
        // the legacy per-leaf searchLeaf path below.
        List<Callable<BatchedPostingVisitor>> fetchTasks = new ArrayList<>(leafReaderContexts.size());
        for (LeafReaderContext context : leafReaderContexts) {
            if (doPrecondition) {
                preconditionQuery(context);
            }
            fetchTasks.add(() -> fetchCentroidIterators(context, filterWeight, knnCollectorManager, visitRatio, leafReaderContexts.size()));
        }
        BatchedPostingVisitor[] perLeafCentroidIterators = taskExecutor.invokeAll(fetchTasks).toArray(BatchedPostingVisitor[]::new);

        TopDocs[] perLeafResults = new TopDocs[leafReaderContexts.size()];

        // Phase 1.5: legacy fallback for non-IVF leaves — run the existing per-leaf searchLeaf
        // path in parallel and stash the resulting TopDocs directly. These leaves don't
        // participate in the global queue.
        List<Integer> fallbackLeaves = new ArrayList<>();
        for (int i = 0; i < leafReaderContexts.size(); i++) {
            if (perLeafCentroidIterators[i] == null) {
                fallbackLeaves.add(i);
            }
        }
        if (fallbackLeaves.isEmpty() == false) {
            List<Callable<TopDocs>> fallbackTasks = new ArrayList<>(fallbackLeaves.size());
            for (int i : fallbackLeaves) {
                LeafReaderContext ctx = leafReaderContexts.get(i);
                fallbackTasks.add(() -> searchLeaf(ctx, filterWeight, knnCollectorManager, visitRatio));
            }
            List<TopDocs> fallbackResults = taskExecutor.invokeAll(fallbackTasks);
            for (int j = 0; j < fallbackLeaves.size(); j++) {
                perLeafResults[fallbackLeaves.get(j)] = fallbackResults.get(j);
            }
        }

        // Phase 2: pair each BatchedPostingVisitor with a per-leaf scorer (owns the collector +
        // a ReentrantLock around the stateful PostingVisitor) and seed a global max-heap with
        // each leaf's initial batch of posting-list tasks.
        LeafPostingsScorer[] postingsScorers = new LeafPostingsScorer[leafReaderContexts.size()];
        PriorityBlockingQueue<PostingSearchTask> taskQueue = new PriorityBlockingQueue<>();
        AtomicInteger outstanding = new AtomicInteger(0);
        for (int i = 0; i < leafReaderContexts.size(); i++) {
            BatchedPostingVisitor bpv = perLeafCentroidIterators[i];
            if (bpv == null) {
                continue;
            }
            LeafReaderContext ctx = leafReaderContexts.get(i);
            IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(visitRatio, numCands, k, knnCollectorManager.longAccumulator);
            AbstractMaxScoreKnnCollector collector = knnCollectorManager.newCollector(Integer.MAX_VALUE, strategy, ctx);
            if (collector == null) {
                continue;
            }
            strategy.setCollector(collector);
            LeafPostingsScorer scorer = new LeafPostingsScorer(i, ctx, bpv, collector);
            postingsScorers[i] = scorer;
            seedBatch(taskQueue, outstanding, scorer);
        }

        // Phase 3: drain the queue with worker callables. Each worker pulls the highest-scoring
        // centroid available across all leaves, executes it under its leaf's lock, and either
        // refills 1-ahead from the same leaf or — when that leaf has fully drained the previous
        // batch — kicks off the next (halved) batch. Workers exit once the queue is empty and
        // nothing else is in flight.
        // TODO: avoid materializing the full filter bitset in ESAcceptDocs.ScorerSupplierAcceptDocs
        // when only used via AcceptDocs.bits() in the parallel path.
        int numWorkers = Math.max(1, leafReaderContexts.size());
        List<Callable<Void>> workers = new ArrayList<>(numWorkers);
        for (int w = 0; w < numWorkers; w++) {
            workers.add(() -> {
                while (true) {
                    PostingSearchTask task = taskQueue.poll();
                    if (task == null) {
                        if (outstanding.get() == 0) {
                            return null;
                        }
                        // Another worker is mid-flight and may refill the queue; brief spin until
                        // it either finishes its refill or the global counter drops to zero.
                        // TODO: replace the spin with a Phaser/CountDownLatch on per-leaf drain.
                        Thread.onSpinWait();
                        continue;
                    }
                    try {
                        if (task.scorer().shouldContinue()) {
                            task.scorer().scorePosting(task.metadata());
                        }
                        int leafInFlight = task.producer().onTaskCompleted();
                        if (leafInFlight == 0 && task.producer().hasNext() && task.scorer().shouldContinue()) {
                            seedBatch(taskQueue, outstanding, task.scorer());
                        }
                    } finally {
                        outstanding.decrementAndGet();
                    }
                }
            });
        }
        taskExecutor.invokeAll(workers);

        // Phase 4: produce per-leaf TopDocs from each IVF scorer's collector (with dedup).
        // Fallback leaves already have their TopDocs populated from Phase 1.5.
        for (int i = 0; i < postingsScorers.length; i++) {
            if (postingsScorers[i] != null) {
                perLeafResults[i] = postingsScorers[i].finalizeTopDocs();
            } else if (perLeafResults[i] == null) {
                perLeafResults[i] = NO_RESULTS;
            }
        }

        TopDocs topK = mergeLeafResults(k, perLeafResults);
        vectorOpsCount = (int) topK.totalHits.value();
        if (topK.scoreDocs.length == 0) {
            return Queries.NO_DOCS_INSTANCE;
        }
        return new KnnScoreDocQuery(topK.scoreDocs, reader);
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

    abstract BatchedPostingVisitor fetchCentroidIterators(
        LeafReaderContext context,
        Weight filterWeight,
        IVFCollectorManager knnCollectorManager,
        float visitRatio,
        int numLeaves
    ) throws IOException;

    /**
     * Pulls the next batch from {@code scorer.bpv()} (size halved from the previous one, floor 1),
     * issues prefetch for each yielded posting list, and enqueues them as {@link PostingSearchTask}
     * instances. Each enqueued task is also counted in {@code outstanding} so workers know the
     * total amount of in-flight work across all leaves.
     */
    private static void seedBatch(PriorityBlockingQueue<PostingSearchTask> queue, AtomicInteger outstanding, LeafPostingsScorer scorer)
        throws IOException {
        BatchedPostingVisitor bpv = scorer.bpv();
        int batch = bpv.nextBatch();
        for (int b = 0; b < batch; b++) {
            PostingMetadata md = bpv.yieldNext();
            if (md == null) {
                break;
            }
            queue.offer(new PostingSearchTask(scorer, bpv, md));
            outstanding.incrementAndGet();
        }
    }

    private TopDocs searchLeaf(LeafReaderContext ctx, Weight filterWeight, IVFCollectorManager knnCollectorManager, float visitRatio)
        throws IOException {
        TopDocs results = getLeafResults(ctx, filterWeight, knnCollectorManager, visitRatio);
        IntObjectHashMap<ScoreDoc> dedupByDoc = new IntObjectHashMap<>(results.scoreDocs.length * 4 / 3);
        for (ScoreDoc scoreDoc : results.scoreDocs) {
            int globalDoc = scoreDoc.doc + ctx.docBase;
            if (dedupByDoc.containsKey(globalDoc) == false) {
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

    TopDocs getLeafResults(LeafReaderContext ctx, Weight filterWeight, IVFCollectorManager knnCollectorManager, float visitRatio)
        throws IOException {
        final LeafReader reader = ctx.reader();
        final Bits liveDocs = reader.getLiveDocs();
        final int maxDoc = reader.maxDoc();

        if (filterWeight == null) {
            return approximateSearch(
                ctx,
                liveDocs == null ? new ESAcceptDocs.ESAcceptDocsAll() : new ESAcceptDocs.BitsAcceptDocs(liveDocs, maxDoc),
                Integer.MAX_VALUE,
                knnCollectorManager,
                visitRatio
            );
        }

        ScorerSupplier supplier = filterWeight.scorerSupplier(ctx);
        if (supplier == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }

        return approximateSearch(
            ctx,
            new ESAcceptDocs.ScorerSupplierAcceptDocs(supplier, liveDocs, maxDoc),
            Integer.MAX_VALUE,
            knnCollectorManager,
            visitRatio
        );
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
