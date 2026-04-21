/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import com.carrotsearch.hppc.IntHashSet;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.DocIdSetIterator;
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
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.util.Bits;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.PostingMetadata;
import org.elasticsearch.search.profile.query.QueryProfiler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.concurrent.Callable;
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

        IVFCollectorManager knnCollectorManager = getKnnCollectorManager(Math.round(2f * k), indexSearcher);
        TaskExecutor taskExecutor = indexSearcher.getTaskExecutor();
        List<LeafReaderContext> leafReaderContexts = reader.leaves();
        final float visitRatio = providedVisitRatio;

        // Parallel posting scoring: gather doc IDs first, then score in parallel across segments.
        // Falls back to legacy single-phase path for codecs that don't support gatherDocIds (ES920).
        if (supportsParallelScoring(leafReaderContexts)) {
            return rewriteParallel(reader, filterWeight, knnCollectorManager, taskExecutor, leafReaderContexts, visitRatio);
        }
        return rewriteLegacy(reader, filterWeight, knnCollectorManager, taskExecutor, leafReaderContexts, visitRatio);
    }

    /**
     * Parallel posting list scoring: two-phase approach that works with or without filters.
     *
     * Phase 1 (per-segment, parallel): visit centroids, gather doc IDs from posting lists,
     *   prefetch posting list data for scoring.
     * Phase 2 (cross-segment, parallel): if filter exists, walk global doc ID queue with
     *   advance() to determine matches; then score matching posting lists from a global work
     *   queue balanced across threads. Shared competitive scores enable early termination.
     */
    private Query rewriteParallel(
        IndexReader reader,
        Weight filterWeight,
        IVFCollectorManager knnCollectorManager,
        TaskExecutor taskExecutor,
        List<LeafReaderContext> leafReaderContexts,
        float visitRatio
    ) throws IOException {
        // ── Phase 1: per-segment gather + filter (streaming per-segment heaps) ──
        List<Callable<List<ScoringItem>>> gatherTasks = new ArrayList<>(leafReaderContexts.size());
        for (LeafReaderContext context : leafReaderContexts) {
            if (doPrecondition) {
                preconditionQuery(context);
            }
            gatherTasks.add(() -> gatherLeaf(context, filterWeight, knnCollectorManager, visitRatio));
        }
        List<List<ScoringItem>> perSegmentItems = taskExecutor.invokeAll(gatherTasks);

        // ── Phase 2: collect global work queue and score in parallel ──
        List<ScoringItem> workQueue = new ArrayList<>();
        for (List<ScoringItem> segmentItems : perSegmentItems) {
            workQueue.addAll(segmentItems);
        }
        if (workQueue.isEmpty()) {
            return Queries.NO_DOCS_INSTANCE;
        }

        // Partition work evenly across threads (preserving segment locality)
        int numThreads = Math.max(1, leafReaderContexts.size());
        List<List<ScoringItem>> partitions = partitionWork(workQueue, numThreads);

        // Score partitions in parallel with shared competitive scores
        List<Callable<TopDocs>> scoringTasks = new ArrayList<>(partitions.size());
        for (List<ScoringItem> partition : partitions) {
            scoringTasks.add(() -> scorePartition(partition, knnCollectorManager));
        }
        TopDocs[] partitionResults = taskExecutor.invokeAll(scoringTasks).toArray(TopDocs[]::new);

        TopDocs topK = TopDocs.merge(k, partitionResults);
        vectorOpsCount = (int) topK.totalHits.value();
        if (topK.scoreDocs.length == 0) {
            return Queries.NO_DOCS_INSTANCE;
        }
        return new KnnScoreDocQuery(topK.scoreDocs, reader);
    }

    /** Legacy single-phase path: one thread per segment, centroid selection + scoring coupled. */
    private Query rewriteLegacy(
        IndexReader reader,
        Weight filterWeight,
        IVFCollectorManager knnCollectorManager,
        TaskExecutor taskExecutor,
        List<LeafReaderContext> leafReaderContexts,
        float visitRatio
    ) throws IOException {
        List<Callable<TopDocs>> tasks = new ArrayList<>(leafReaderContexts.size());
        for (LeafReaderContext context : leafReaderContexts) {
            if (doPrecondition) {
                preconditionQuery(context);
            }
            tasks.add(() -> searchLeaf(context, filterWeight, knnCollectorManager, visitRatio));
        }
        TopDocs[] perLeafResults = taskExecutor.invokeAll(tasks).toArray(TopDocs[]::new);

        TopDocs topK = TopDocs.merge(k, perLeafResults);
        vectorOpsCount = (int) topK.totalHits.value();
        if (topK.scoreDocs.length == 0) {
            return Queries.NO_DOCS_INSTANCE;
        }
        return new KnnScoreDocQuery(topK.scoreDocs, reader);
    }

    private TopDocs searchLeaf(LeafReaderContext ctx, Weight filterWeight, IVFCollectorManager knnCollectorManager, float visitRatio)
        throws IOException {
        TopDocs results = getLeafResults(ctx, filterWeight, knnCollectorManager, visitRatio);
        return deduplicateAndGlobalize(results, ctx);
    }

    private static TopDocs deduplicateAndGlobalize(TopDocs results, LeafReaderContext ctx) {
        IntHashSet dedup = new IntHashSet(results.scoreDocs.length * 4 / 3);
        int deduplicateCount = 0;
        for (ScoreDoc scoreDoc : results.scoreDocs) {
            if (dedup.add(scoreDoc.doc)) {
                deduplicateCount++;
            }
        }
        ScoreDoc[] deduplicatedScoreDocs = new ScoreDoc[deduplicateCount];
        dedup.clear();
        int index = 0;
        for (ScoreDoc scoreDoc : results.scoreDocs) {
            if (dedup.add(scoreDoc.doc)) {
                scoreDoc.doc += ctx.docBase;
                deduplicatedScoreDocs[index++] = scoreDoc;
            }
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

    // ── Abstract methods for subclass implementation ──

    /** Whether all segments support the parallel gather+score path (ES940+ codecs). */
    abstract boolean supportsParallelScoring(List<LeafReaderContext> leafReaderContexts);

    abstract void preconditionQuery(LeafReaderContext context) throws IOException;

    abstract TopDocs approximateSearch(
        LeafReaderContext context,
        AcceptDocs acceptDocs,
        int visitedLimit,
        IVFCollectorManager knnCollectorManager,
        float visitRatio
    ) throws IOException;

    /**
     * Phase 1: per-segment centroid selection + doc ID gathering via streaming min-heap.
     * If filter: walks gathered doc IDs in sorted order with advance() to build per-PL match lists.
     * If no filter: all gathered PLs become ScoringItems directly (matchedDocIds=null).
     * Prefetches posting list data as PLs are consumed from the heap.
     */
    abstract List<ScoringItem> gatherLeaf(
        LeafReaderContext context,
        Weight filterWeight,
        IVFCollectorManager knnCollectorManager,
        float visitRatio
    ) throws IOException;

    /** Phase 2: score a partition of posting lists, potentially from multiple segments. */
    abstract TopDocs scorePartition(List<ScoringItem> items, IVFCollectorManager knnCollectorManager) throws IOException;

    protected IVFCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
        return new IVFCollectorManager(k, searcher);
    }

    @Override
    public final void profile(QueryProfiler queryProfiler) {
        queryProfiler.addVectorOpsCount(vectorOpsCount);
    }

    // ── Work partitioning ──

    /** Partition scoring items into balanced chunks for parallel execution. */
    static List<List<ScoringItem>> partitionWork(List<ScoringItem> items, int numThreads) {
        long totalWork = 0;
        for (ScoringItem item : items) {
            totalWork += item.numDocs;
        }
        long targetPerThread = Math.max(1, totalWork / numThreads);
        List<List<ScoringItem>> partitions = new ArrayList<>();
        List<ScoringItem> current = new ArrayList<>();
        long currentWork = 0;
        for (ScoringItem item : items) {
            current.add(item);
            currentWork += item.numDocs;
            if (currentWork >= targetPerThread && partitions.size() < numThreads - 1) {
                partitions.add(current);
                current = new ArrayList<>();
                currentWork = 0;
            }
        }
        if (current.isEmpty() == false) {
            partitions.add(current);
        }
        return partitions;
    }

    // ── Filter intersection via min-heap merge ──

    /**
     * Processes gathered posting lists through a per-segment min-heap, checking each doc ID
     * against the filter via advance(). Emits ScoringItems as PLs are fully consumed.
     * SOAR: same docId from multiple PLs is consecutive in the heap — advance() handles naturally.
     */
    static List<ScoringItem> gatherAndFilter(
        LeafReaderContext leafCtx,
        List<IVFVectorsReader.GatheredPostingList> gathered,
        DocIdSetIterator filterIterator,
        IVFVectorsReader<?> ivfReader,
        String field
    ) throws IOException {
        PriorityQueue<HeapEntry> heap = new PriorityQueue<>();
        for (IVFVectorsReader.GatheredPostingList pl : gathered) {
            if (pl.numDocs() > 0) {
                heap.add(new HeapEntry(pl.metadata(), pl.docIds(), 0, pl.numDocs()));
            }
        }

        // Per-PL match accumulation (keyed by PostingMetadata identity)
        java.util.IdentityHashMap<PostingMetadata, List<Integer>> perPLMatches = new java.util.IdentityHashMap<>();

        List<ScoringItem> result = new ArrayList<>();

        while (heap.isEmpty() == false) {
            HeapEntry top = heap.poll();
            int docId = top.currentDocId();

            // Filter check — iterator only advances forward
            if (filterIterator.docID() <= docId) {
                filterIterator.advance(docId);
            }
            if (filterIterator.docID() == docId) {
                perPLMatches.computeIfAbsent(top.metadata, m -> new ArrayList<>()).add(docId);
            }

            if (top.advance()) {
                heap.add(top);
            } else {
                // PL fully consumed — emit ScoringItem if it has matches
                List<Integer> matches = perPLMatches.remove(top.metadata);
                if (matches != null && matches.isEmpty() == false) {
                    int[] matchArray = matches.stream().mapToInt(Integer::intValue).toArray();
                    result.add(new ScoringItem(leafCtx, top.metadata, matchArray, matchArray.length));
                    ivfReader.prefetchPostingList(field, top.metadata);
                }
            }
        }
        return result;
    }

    // ── Data types ──

    /**
     * A posting list to score: metadata to locate it, and the doc IDs that should be scored.
     * When no filter: matchedDocIds is null and numDocs is the full posting list size.
     * When filtered: matchedDocIds contains only the docs that passed the filter.
     */
    record ScoringItem(LeafReaderContext leafCtx, PostingMetadata metadata, int[] matchedDocIds, int numDocs) {}

    static class HeapEntry implements Comparable<HeapEntry> {
        final PostingMetadata metadata;
        final int[] docIds;
        final int limit;
        int cursor;

        HeapEntry(PostingMetadata metadata, int[] docIds, int cursor, int limit) {
            this.metadata = metadata;
            this.docIds = docIds;
            this.cursor = cursor;
            this.limit = limit;
        }

        int currentDocId() {
            return docIds[cursor];
        }

        boolean advance() {
            cursor++;
            return cursor < limit;
        }

        @Override
        public int compareTo(HeapEntry other) {
            return Integer.compare(currentDocId(), other.currentDocId());
        }
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
