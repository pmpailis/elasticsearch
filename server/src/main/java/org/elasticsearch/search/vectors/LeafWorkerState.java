/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;

/**
 * Per-worker, per-leaf scoring state for the parallel IVF posting-list search pipeline in
 * {@link AbstractIVFKnnVectorQuery}.
 * <p>
 * Posting-list work is distributed across a fixed pool of workers, and a single leaf may be scored
 * concurrently by several workers. Each worker builds its own instance for each leaf it touches and
 * never shares it. The clusters file handle is a clone of the reader's shared handle — clones share
 * the same virtual address mappings (warm TLB entries) while giving each worker its own seek position.
 * The shared competitive frontier lives in the {@link IVFKnnSearchStrategy}'s accumulator and is the
 * only state intentionally shared between workers.
 */
final class LeafWorkerState {

    final IVFVectorsReader.PostingVisitor visitor;
    final AbstractMaxScoreKnnCollector collector;
    final IVFKnnSearchStrategy strategy;

    LeafWorkerState(IVFVectorsReader.PostingVisitor visitor, AbstractMaxScoreKnnCollector collector, IVFKnnSearchStrategy strategy) {
        this.visitor = visitor;
        this.collector = collector;
        this.strategy = strategy;
    }
}
