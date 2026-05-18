/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.elasticsearch.index.codec.vectors.diskbbq.BatchedPostingVisitor;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.PostingMetadata;

import java.io.IOException;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Per-leaf executor for posting-list scoring in the parallel IVF search path.
 *
 * <p>Owns the leaf's {@link IVFVectorsReader.PostingVisitor} (single-use, stateful) and its
 * {@link AbstractMaxScoreKnnCollector}, and serializes their use behind a {@link ReentrantLock}
 * so worker threads pulling tasks from the global queue can score posting lists belonging to
 * this leaf without stepping on each other. Cross-leaf parallelism is the win; within-leaf
 * remains serial.
 *
 * <p>Stop accounting mirrors the per-leaf two-phase logic in {@code IVFVectorsReader.search()}:
 * phase 1 visits postings until {@code maxVectorVisited} is exceeded and a competitive score
 * has been seen; phase 2 (only when filtered) keeps visiting until the actual scored doc count
 * reaches an expected filter-aware floor.
 */
final class LeafPostingsScorer {

    private final int leafIdx;
    private final LeafReaderContext ctx;
    private final BatchedPostingVisitor bpv;
    private final AbstractMaxScoreKnnCollector collector;
    private final ReentrantLock lock = new ReentrantLock();

    private long expectedDocs = 0;
    private long actualDocs = 0;

    LeafPostingsScorer(int leafIdx, LeafReaderContext ctx, BatchedPostingVisitor bpv, AbstractMaxScoreKnnCollector collector) {
        this.leafIdx = leafIdx;
        this.ctx = ctx;
        this.bpv = bpv;
        this.collector = collector;
    }

    int leafIdx() {
        return leafIdx;
    }

    BatchedPostingVisitor bpv() {
        return bpv;
    }

    /**
     * Returns {@code true} when this leaf is still competitive — i.e. a task pulled from the
     * queue is still worth scoring, and the orchestrator should keep refilling from this leaf if
     * more posting lists remain.
     *
     * <p>Phase 1: continue until {@code maxVectorVisited} is exceeded AND we have at least one
     * competitive score recorded by the collector. Phase 2 (only when a filter is applied):
     * continue until {@code actualDocs} reaches the filter-aware expected floor or at least
     * {@code k} docs have been scored.
     *
     * <p>This intentionally does not check {@code bpv.hasNext()}: tasks already yielded into the
     * queue must still be scored even after the underlying iterator is exhausted. The refill
     * decision separately combines this method's result with {@code bpv.hasNext()}.
     */
    boolean shouldContinue() {
        lock.lock();
        try {
            if (bpv.maxVectorVisited() > expectedDocs || collector.minCompetitiveSimilarity() == Float.NEGATIVE_INFINITY) {
                return true;
            }
            if (bpv.filtered()) {
                float unfilteredRatioVisited = (float) expectedDocs / bpv.numVectors();
                int filteredVectors = (int) Math.ceil(bpv.numVectors() * bpv.percentFiltered());
                float expectedScored = Math.min(2f * filteredVectors * unfilteredRatioVisited, expectedDocs / 2f);
                return actualDocs < expectedScored || actualDocs < collector.k();
            }
            return false;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Executes one posting list under the leaf's lock: resets the visitor onto the given
     * posting metadata, scores its docs into the collector, and ticks the strategy's
     * cross-leaf {@code nextVectorsBlock} broadcast.
     */
    void scorePosting(PostingMetadata meta) throws IOException {
        lock.lock();
        try {
            expectedDocs += bpv.postingVisitor().resetPostingsScorer(meta);
            actualDocs += bpv.postingVisitor().visit(collector);
            if (collector.getSearchStrategy() != null) {
                collector.getSearchStrategy().nextVectorsBlock();
            }
        } finally {
            lock.unlock();
        }
    }

    /**
     * Produces the per-leaf {@link TopDocs}: pulls results from the collector (preferring the
     * bulk path when available) and deduplicates by global doc id, mirroring the legacy
     * {@code AbstractIVFKnnVectorQuery.searchLeaf} behavior.
     */
    TopDocs finalizeTopDocs() {
        TopDocs results = collector instanceof BulkKnnCollector bulk ? bulk.unsortedTopK() : collector.topDocs();
        if (results == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }
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
}
