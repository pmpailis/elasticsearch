/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.util.FixedBitSet;
import org.elasticsearch.index.codec.vectors.cluster.NeighborQueue;
import org.elasticsearch.test.ESTestCase;

import java.util.concurrent.atomic.LongAccumulator;

public class IVFKnnSearchStrategyTests extends ESTestCase {

    public void testMaxScorePropagation() {
        LongAccumulator accumulator = new LongAccumulator(Long::max, AbstractMaxScoreKnnCollector.LEAST_COMPETITIVE);
        IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(0.5f, 100, 10, accumulator);
        MaxScoreTopKnnCollector collector = new MaxScoreTopKnnCollector(2, 1000, strategy);
        strategy.setCollector(collector);

        collector.collect(1, 0.9f);

        // queue is not saturated, it should not be updated
        strategy.nextVectorsBlock();
        assertEquals(AbstractMaxScoreKnnCollector.LEAST_COMPETITIVE, accumulator.get());
        assertEquals(AbstractMaxScoreKnnCollector.LEAST_COMPETITIVE, collector.getMinCompetitiveDocScore());

        // accumulator should now be updated
        collector.collect(2, 0.9f);
        long competitiveScore = NeighborQueue.encodeRaw(2, 0.9f);
        strategy.nextVectorsBlock();
        assertEquals(competitiveScore, accumulator.get());
        assertEquals(competitiveScore, collector.getMinCompetitiveDocScore());

        // updated accumulator directly with more competitive score
        competitiveScore = NeighborQueue.encodeRaw(3, 1.5f);
        accumulator.accumulate(competitiveScore);
        assertEquals(competitiveScore, accumulator.get());
        strategy.nextVectorsBlock();
        assertEquals(competitiveScore, collector.getMinCompetitiveDocScore());
        assertEquals(competitiveScore, accumulator.get());
    }

    /**
     * A diversifying collector whose heap is not full must not export its heap top as a competitive floor: an
     * under-filled heap's top is just the worst parent seen so far, not a bound on the k-th best. Before the
     * {@code heap.size() >= k()} guards, the early publish below poisoned the accumulator and, once folded back,
     * permanently inflated the collector's own pruning floor.
     */
    public void testDiversifiedMaxScorePropagationRequiresFullHeap() {
        LongAccumulator accumulator = new LongAccumulator(Long::max, AbstractMaxScoreKnnCollector.LEAST_COMPETITIVE);
        IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(0.5f, 100, 10, accumulator);
        FixedBitSet parentBitSet = new FixedBitSet(100);
        parentBitSet.set(10);
        parentBitSet.set(21);
        DiversifyingNearestChildrenKnnCollector collector = new DiversifyingNearestChildrenKnnCollector(2, 1000, strategy, parentBitSet);
        strategy.setCollector(collector);

        // one parent collected at a high score: the heap is not full, so nothing may be published or folded back
        collector.collect(9, 0.9f);
        strategy.nextVectorsBlock();
        assertEquals(AbstractMaxScoreKnnCollector.LEAST_COMPETITIVE, accumulator.get());
        assertEquals(AbstractMaxScoreKnnCollector.LEAST_COMPETITIVE, collector.getMinCompetitiveDocScore());
        assertEquals(Float.NEGATIVE_INFINITY, collector.minCompetitiveSimilarity(), 0.0f);

        // a second, distinct parent at a lower score fills the heap: its top is now a valid floor
        collector.collect(20, 0.2f);
        strategy.nextVectorsBlock();
        assertEquals(NeighborQueue.encodeRaw(20, 0.2f), accumulator.get());
        // the earlier 0.9 collection must not have poisoned the pruning floor
        assertEquals(0.2f, collector.minCompetitiveSimilarity(), 0.0f);
    }
}
