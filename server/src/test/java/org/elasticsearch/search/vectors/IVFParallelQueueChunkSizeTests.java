/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.elasticsearch.test.ESTestCase;

/**
 * Unit tests for {@link AbstractIVFKnnVectorQuery#chunkSize(int, int)}, the per-claim sizing of the global
 * posting-list queue. The chunk size must overlap I/O (a claim covers a run a worker can prefetch ahead of
 * scoring) without sacrificing the queue's fine-grained load balancing (there must remain at least as many
 * chunks as workers so no worker is left idle).
 */
public class IVFParallelQueueChunkSizeTests extends ESTestCase {

    public void testSingleWorkerScoresSequentially() {
        // With a single worker (e.g. a saturated search pool) there is no balancing to preserve, so the whole
        // queue is drained one posting at a time.
        assertEquals(1, AbstractIVFKnnVectorQuery.chunkSize(1, 1));
        assertEquals(1, AbstractIVFKnnVectorQuery.chunkSize(1_000, 1));
    }

    public void testFewerPostingsThanWorkersFallsBackToOne() {
        // Each worker can get at most one posting, so chunking would only starve workers.
        assertEquals(1, AbstractIVFKnnVectorQuery.chunkSize(4, 8));
        assertEquals(1, AbstractIVFKnnVectorQuery.chunkSize(8, 8));
    }

    public void testChunkNeverExceedsAFairPerWorkerShare() {
        // chunk <= total/workers guarantees there are at least `workers` chunks, so every worker can claim one.
        for (int workers = 2; workers <= 16; workers++) {
            for (int total = workers + 1; total <= 100_000; total *= 7) {
                int chunk = AbstractIVFKnnVectorQuery.chunkSize(total, workers);
                assertTrue("chunk must be positive", chunk >= 1);
                assertTrue("chunk must not exceed the queue", chunk <= total);
                assertTrue(
                    "chunk [" + chunk + "] for total [" + total + "], workers [" + workers + "] must leave >= workers chunks",
                    chunk <= total / workers
                );
                int chunks = (total + chunk - 1) / chunk;
                assertTrue("must produce at least `workers` chunks", chunks >= workers);
            }
        }
    }

    public void testChunkGrowsWithQueueDepth() {
        // For a fixed worker count, a deeper queue yields larger (or equal) chunks, so prefetch windows widen.
        int workers = 8;
        int small = AbstractIVFKnnVectorQuery.chunkSize(1_000, workers);
        int large = AbstractIVFKnnVectorQuery.chunkSize(1_000_000, workers);
        assertTrue("deeper queue should not shrink the chunk", large >= small);
        assertTrue("deep queue should chunk above 1 to enable prefetch", large > 1);
    }

    public void testDegenerateInputs() {
        assertEquals(1, AbstractIVFKnnVectorQuery.chunkSize(0, 8));
        assertEquals(1, AbstractIVFKnnVectorQuery.chunkSize(0, 0));
        assertEquals(1, AbstractIVFKnnVectorQuery.chunkSize(10, 0));
    }
}
