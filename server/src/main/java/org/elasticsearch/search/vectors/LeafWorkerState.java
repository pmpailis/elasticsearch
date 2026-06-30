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
import org.apache.lucene.util.IOUtils;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;

import java.io.Closeable;
import java.io.IOException;

/**
 * Per-worker, per-leaf scoring state for the parallel IVF posting-list search pipeline in
 * {@link AbstractIVFKnnVectorQuery}.
 * <p>
 * Posting-list work is distributed across a fixed pool of workers, and a single leaf may be scored
 * concurrently by several workers. Each worker builds its own instance for each leaf it touches and
 * never shares it. A leaf scored by a single worker simply {@link IndexInput#clone() clones} the reader's
 * shared handles — clones share the same virtual address mappings (warm TLB entries) while giving each
 * worker its own seek position. A leaf that may be scored concurrently by several workers instead opens
 * thread-confined handles ({@link IVFVectorsReader#openLightweightClusters()} /
 * {@link IVFVectorsReader#openLightweightCentroids()}) so reads avoid the cross-thread CAS on the shared
 * {@code MemorySegment} session. Either way the handles are owned by this state and must be
 * {@link #close() closed} on the worker thread that opened them. The shared competitive frontier lives in
 * the {@link IVFKnnSearchStrategy}'s accumulator and is the only state intentionally shared between workers.
 */
final class LeafWorkerState implements Closeable {

    final IndexInput clustersHandle;
    final IndexInput centroidsHandle;
    final IVFVectorsReader.PostingVisitor visitor;
    final AbstractMaxScoreKnnCollector collector;
    final IVFKnnSearchStrategy strategy;

    LeafWorkerState(
        IndexInput clustersHandle,
        IndexInput centroidsHandle,
        IVFVectorsReader.PostingVisitor visitor,
        AbstractMaxScoreKnnCollector collector,
        IVFKnnSearchStrategy strategy
    ) {
        this.clustersHandle = clustersHandle;
        this.centroidsHandle = centroidsHandle;
        this.visitor = visitor;
        this.collector = collector;
        this.strategy = strategy;
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(clustersHandle, centroidsHandle);
    }
}
