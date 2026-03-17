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
 * A KNN collector that scores all documents without the filter (preserving SIMD bulk scoring)
 * and applies the filter only to the final result set. Used when filter selectivity is high
 * (most docs pass), where passing the filter as AcceptDocs would hurt SIMD performance.
 *
 * <p>Uses normal competitive pruning on the oversampled queue — the larger queue size naturally
 * lowers the competitive threshold, causing the codec to visit more centroids to compensate
 * for post-filtering loss. The filter is applied only in {@link #topDocs()}.
 */
class PostFilteringKnnCollector extends AbstractMaxScoreKnnCollector {

    private final int originalK;
    private final NeighborQueue queue;
    private final Weight filterWeight;
    private final LeafReaderContext leafContext;
    private long minCompetitiveDocScore = LEAST_COMPETITIVE;
    private float minCompetitiveSimilarity = Float.NEGATIVE_INFINITY;

    PostFilteringKnnCollector(
        int oversampledK,
        int originalK,
        long visitLimit,
        KnnSearchStrategy searchStrategy,
        Weight filterWeight,
        LeafReaderContext leafContext
    ) {
        super(oversampledK, visitLimit, searchStrategy);
        this.originalK = originalK;
        this.queue = new NeighborQueue(oversampledK, false);
        this.filterWeight = filterWeight;
        this.leafContext = leafContext;
    }

    @Override
    public boolean collect(int docId, float similarity) {
        return queue.insertWithOverflow(docId, similarity);
    }

    @Override
    public float minCompetitiveSimilarity() {
        return queue.size() < k() ? Float.NEGATIVE_INFINITY : Math.max(minCompetitiveSimilarity, queue.topScore());
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
        this.minCompetitiveSimilarity = NeighborQueue.decodeScoreRaw(this.minCompetitiveDocScore);
    }

    @Override
    public TopDocs topDocs() {
        int size = queue.size();
        // Drain queue into descending score order (pop min-heap from end)
        int[] docIds = new int[size];
        float[] scores = new float[size];
        for (int i = size - 1; i >= 0; i--) {
            docIds[i] = queue.topNode();
            scores[i] = queue.topScore();
            queue.pop();
        }

        // Sort a copy ascending by docId for efficient advance()
        int[] sortedDocIds = docIds.clone();
        Arrays.sort(sortedDocIds);

        // Get fresh filter iterator and find passing docs
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
            throw new  UncheckedIOException(e);
        }

        // Rebuild ScoreDocs from descending score order, keeping only passing docs, capped at originalK
        List<ScoreDoc> result = new ArrayList<>();
        for (int i = 0; i < size && result.size() < originalK; i++) {
            if (passing.contains(docIds[i])) {
                result.add(new ScoreDoc(docIds[i], scores[i]));
            }
        }

        TotalHits.Relation relation = earlyTerminated() ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO;
        return new TopDocs(new TotalHits(visitedCount(), relation), result.toArray(new ScoreDoc[0]));
    }

    private DocIdSetIterator newFilterIterator() throws IOException {
        ScorerSupplier ss = filterWeight.scorerSupplier(leafContext);
        if (ss == null) {
            return DocIdSetIterator.empty();
        }
        return ss.get(NO_MORE_DOCS).iterator();
    }
}
