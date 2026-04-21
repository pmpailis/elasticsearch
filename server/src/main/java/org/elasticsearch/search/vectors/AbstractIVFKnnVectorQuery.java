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

        // Use two-phase path when a filter is present and segments support it
        if (filterWeight != null && supportsTwoPhaseSearch(leafReaderContexts)) {
            return rewriteTwoPhase(indexSearcher, reader, filterWeight, knnCollectorManager, taskExecutor, leafReaderContexts, visitRatio);
        }

        return rewriteSinglePhase(reader, filterWeight, knnCollectorManager, taskExecutor, leafReaderContexts, visitRatio);
    }

    private Query rewriteSinglePhase(
        IndexReader reader,
        Weight filterWeight,
        IVFCollectorManager knnCollectorManager,
        TaskExecutor taskExecutor,
        List<LeafReaderContext> leafReaderContexts,
        float visitRatio
    ) throws IOException {
        List<Callable<TopDocs>> tasks = new ArrayList<>(leafReaderContexts.size());
        for (LeafReaderContext context : leafReaderContexts) {
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

    /**
     * Two-phase search: Phase 1 gathers doc IDs + filters per segment, Phase 2 scores
     * matching posting lists in parallel from a global work queue.
     */
    private Query rewriteTwoPhase(
        IndexSearcher indexSearcher,
        IndexReader reader,
        Weight filterWeight,
        IVFCollectorManager knnCollectorManager,
        TaskExecutor taskExecutor,
        List<LeafReaderContext> leafReaderContexts,
        float visitRatio
    ) throws IOException {
        // ── Phase 1: per-segment gather + filter ──
        List<Callable<SegmentGatherResult>> gatherTasks = new ArrayList<>(leafReaderContexts.size());
        for (LeafReaderContext context : leafReaderContexts) {
            if (doPrecondition) {
                preconditionQuery(context);
            }
            gatherTasks.add(() -> gatherLeaf(context, filterWeight, knnCollectorManager, visitRatio));
        }
        List<SegmentGatherResult> gatherResults = taskExecutor.invokeAll(gatherTasks);

        // ── Phase 2: cross-segment parallel scoring ──
        List<ScoringItem> allItems = new ArrayList<>();
        for (SegmentGatherResult sgr : gatherResults) {
            allItems.addAll(sgr.scoringItems);
        }
        if (allItems.isEmpty()) {
            return Queries.NO_DOCS_INSTANCE;
        }

        int numThreads = Math.max(1, leafReaderContexts.size());
        List<List<ScoringItem>> partitions = partitionWork(allItems, numThreads);

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

    private TopDocs searchLeaf(LeafReaderContext ctx, Weight filterWeight, IVFCollectorManager knnCollectorManager, float visitRatio)
        throws IOException {
        TopDocs results = getLeafResults(ctx, filterWeight, knnCollectorManager, visitRatio);
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

    /**
     * Returns true if all segments support the two-phase gather+score path.
     * Checks that the codec reader supports gatherDocIds() (ES940+ only, not ES920).
     */
    abstract boolean supportsTwoPhaseSearch(List<LeafReaderContext> leafReaderContexts);

    abstract void preconditionQuery(LeafReaderContext context) throws IOException;

    abstract TopDocs approximateSearch(
        LeafReaderContext context,
        AcceptDocs acceptDocs,
        int visitedLimit,
        IVFCollectorManager knnCollectorManager,
        float visitRatio
    ) throws IOException;

    /**
     * Phase 1: gather doc IDs from posting lists without scoring, then filter via advance().
     * Returns per-PL match lists as ScoringItems.
     */
    abstract SegmentGatherResult gatherLeaf(
        LeafReaderContext context,
        Weight filterWeight,
        IVFCollectorManager knnCollectorManager,
        float visitRatio
    ) throws IOException;

    /**
     * Phase 2: score a partition of ScoringItems from potentially multiple segments.
     */
    abstract TopDocs scorePartition(List<ScoringItem> items, IVFCollectorManager knnCollectorManager) throws IOException;

    protected IVFCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
        return new IVFCollectorManager(k, searcher);
    }

    @Override
    public final void profile(QueryProfiler queryProfiler) {
        queryProfiler.addVectorOpsCount(vectorOpsCount);
    }

    // ── Work partitioning ──

    static List<List<ScoringItem>> partitionWork(List<ScoringItem> items, int numThreads) {
        long totalWork = 0;
        for (ScoringItem item : items) {
            totalWork += item.matchCount;
        }
        long targetPerThread = Math.max(1, totalWork / numThreads);
        List<List<ScoringItem>> partitions = new ArrayList<>();
        List<ScoringItem> current = new ArrayList<>();
        long currentWork = 0;
        for (ScoringItem item : items) {
            current.add(item);
            currentWork += item.matchCount;
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
     * Merges doc IDs from multiple gathered posting lists via min-heap, walks a filter iterator
     * with advance(), and builds per-PL match lists. SOAR duplicates are handled naturally
     * since all occurrences of the same docId are consecutive in the heap.
     */
    static List<ScoringItem> mergeAndFilter(
        LeafReaderContext leafCtx,
        List<IVFVectorsReader.GatheredPostingList> gathered,
        DocIdSetIterator filterIterator
    ) throws IOException {
        PriorityQueue<HeapEntry> heap = new PriorityQueue<>();
        for (int p = 0; p < gathered.size(); p++) {
            IVFVectorsReader.GatheredPostingList pl = gathered.get(p);
            if (pl.numDocs() > 0) {
                heap.add(new HeapEntry(p, pl.docIds(), 0, pl.numDocs()));
            }
        }

        @SuppressWarnings("unchecked")
        List<Integer>[] plMatchLists = new List[gathered.size()];

        while (heap.isEmpty() == false) {
            HeapEntry top = heap.poll();
            int docId = top.currentDocId();
            int plIdx = top.plIndex;

            // Filter check — iterator only advances forward
            if (filterIterator.docID() <= docId) {
                filterIterator.advance(docId);
            }
            if (filterIterator.docID() == docId) {
                if (plMatchLists[plIdx] == null) {
                    plMatchLists[plIdx] = new ArrayList<>();
                }
                plMatchLists[plIdx].add(docId);
            }

            if (top.advance()) {
                heap.add(top);
            }
        }

        // Build ScoringItems and prefetch matching posting lists
        List<ScoringItem> result = new ArrayList<>();
        for (int p = 0; p < gathered.size(); p++) {
            if (plMatchLists[p] == null || plMatchLists[p].isEmpty()) {
                continue;
            }
            int[] matchArray = plMatchLists[p].stream().mapToInt(Integer::intValue).toArray();
            result.add(new ScoringItem(leafCtx, gathered.get(p).metadata(), matchArray, matchArray.length));
        }
        return result;
    }

    // ── Data types ──

    record ScoringItem(LeafReaderContext leafCtx, PostingMetadata metadata, int[] matchedDocIds, int matchCount) {}

    record SegmentGatherResult(LeafReaderContext leafCtx, List<ScoringItem> scoringItems) {}

    static class HeapEntry implements Comparable<HeapEntry> {
        final int plIndex;
        final int[] docIds;
        final int limit;
        int cursor;

        HeapEntry(int plIndex, int[] docIds, int cursor, int limit) {
            this.plIndex = plIndex;
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
