/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.store.IndexInput;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;

import java.io.Closeable;
import java.io.IOException;

/**
 * Per-worker, per-leaf scoring state for the parallel IVF posting-list search pipeline in
 * {@link AbstractIVFKnnVectorQuery}.
 * <p>
 * Posting-list work is distributed across a fixed pool of workers, and a single leaf may be scored
 * concurrently by several workers. To avoid cross-thread contention on the memory-mapped arena (see
 * {@link IVFVectorsReader#openLightweightClusters()}), every worker builds its <em>own</em> instance
 * for each leaf it touches and never shares it. The {@link #confinedClusters} handle is backed by a
 * confined arena, so it MUST be opened, used, and closed on the same thread; {@link #close()}
 * releases it on the worker thread once that worker has finished scoring and drained
 * {@link #collector}. The shared competitive frontier lives in the {@link IVFKnnSearchStrategy}'s
 * accumulator and is the only state intentionally shared between workers.
 */
final class LeafWorkerState implements Closeable {

    final IndexInput confinedClusters;
    final IVFVectorsReader.PostingVisitor visitor;
    final AbstractMaxScoreKnnCollector collector;
    final IVFKnnSearchStrategy strategy;

    LeafWorkerState(
        IndexInput confinedClusters,
        IVFVectorsReader.PostingVisitor visitor,
        AbstractMaxScoreKnnCollector collector,
        IVFKnnSearchStrategy strategy
    ) {
        this.confinedClusters = confinedClusters;
        this.visitor = visitor;
        this.collector = collector;
        this.strategy = strategy;
    }

    /**
     * Closes the confined clusters handle. Closing the root handle releases every slice and clone
     * derived from it, so the per-leaf posting-list slice does not need to be closed separately.
     */
    @Override
    public void close() throws IOException {
        confinedClusters.close();
    }
}
