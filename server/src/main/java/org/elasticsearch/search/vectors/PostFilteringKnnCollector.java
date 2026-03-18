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

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.elasticsearch.index.codec.vectors.cluster.NeighborQueue;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * A KNN collector that scores all documents without the filter (preserving SIMD bulk scoring for IVF
 * and avoiding per-visit filter evaluation for HNSW) and applies the filter only at result time.
 *
 * <p>During search, this collector behaves identically to {@code MaxScoreTopKnnCollector}: the queue
 * fills naturally and the competitive threshold tightens monotonically. At {@link #topDocs()} time,
 * the queue is drained once, the filter is applied, and the top {@code originalK} passing results
 * are returned.
 *
 * <p>The queue is oversampled ({@code queueSize > originalK}) to compensate for filtered-out results.
 */
class PostFilteringKnnCollector extends AbstractMaxScoreKnnCollector {

    private final int queueSize;
    private final int originalK;
    private final NeighborQueue queue;
    private final Weight filterWeight;
    private final LeafReaderContext leafContext;

    private long minCompetitiveDocScore = LEAST_COMPETITIVE;
    private float decodedMinCompetitiveSimilarity = Float.NEGATIVE_INFINITY;

    PostFilteringKnnCollector(
        int queueSize,
        int originalK,
        long visitLimit,
        KnnSearchStrategy searchStrategy,
        Weight filterWeight,
        LeafReaderContext leafContext
    ) {
        super(queueSize, visitLimit, searchStrategy);
        this.queueSize = queueSize;
        this.originalK = originalK;
        this.queue = new NeighborQueue(queueSize, false);
        this.filterWeight = filterWeight;
        this.leafContext = leafContext;
    }

    @Override
    public boolean collect(int docId, float similarity) {
        return queue.insertWithOverflow(docId, similarity);
    }

    @Override
    public float minCompetitiveSimilarity() {
        return queue.size() >= k() ? Math.max(decodedMinCompetitiveSimilarity, queue.topScore()) : decodedMinCompetitiveSimilarity;
    }

    @Override
    public int numCollected() {
        return queue.size();
    }

    @Override
    public long getMinCompetitiveDocScore() {
        return queue.size() > 0 ? Math.max(minCompetitiveDocScore, queue.peek()) : minCompetitiveDocScore;
    }

    @Override
    void updateMinCompetitiveDocScore(long minCompetitiveDocScore) {
        long queueMinCompetitiveDocScore = queue.size() > 0 ? queue.peek() : LEAST_COMPETITIVE;
        this.minCompetitiveDocScore = Math.max(this.minCompetitiveDocScore, Math.max(queueMinCompetitiveDocScore, minCompetitiveDocScore));
        this.decodedMinCompetitiveSimilarity = NeighborQueue.decodeScoreRaw(this.minCompetitiveDocScore);
    }

    @Override
    public TopDocs topDocs() {
        int size = queue.size();
        // Drain queue in score order (highest first)
        int[] docIds = new int[size];
        float[] scores = new float[size];
        for (int i = size - 1; i >= 0; i--) {
            docIds[i] = queue.topNode();
            scores[i] = queue.topScore();
            queue.pop();
        }
        // Apply filter, keep top originalK passing results
        IntHashSet passing = findPassingDocs(docIds);
        List<ScoreDoc> result = new ArrayList<>();
        for (int i = 0; i < size && result.size() < originalK; i++) {
            if (passing.contains(docIds[i])) {
                result.add(new ScoreDoc(docIds[i], scores[i]));
            }
        }
        TotalHits.Relation relation = earlyTerminated() ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO;
        return new TopDocs(new TotalHits(visitedCount(), relation), result.toArray(new ScoreDoc[0]));
    }

    private IntHashSet findPassingDocs(int[] docIds) {
        int[] sortedDocIds = docIds.clone();
        Arrays.sort(sortedDocIds);

        IntHashSet passing = new IntHashSet();
        try {
            DocIdSetIterator filterIterator = newFilterIterator();
            int filterDoc = -1;
            for (int sortedDocId : sortedDocIds) {
                if (filterDoc < sortedDocId) {
                    filterDoc = filterIterator.advance(sortedDocId);
                }
                if (filterDoc == sortedDocId) {
                    passing.add(sortedDocId);
                }
                if (filterDoc == NO_MORE_DOCS) {
                    break;
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return passing;
    }

    private DocIdSetIterator newFilterIterator() throws IOException {
        ScorerSupplier ss = filterWeight.scorerSupplier(leafContext);
        if (ss == null) {
            return DocIdSetIterator.empty();
        }
        return ss.get(NO_MORE_DOCS).iterator();
    }
}
