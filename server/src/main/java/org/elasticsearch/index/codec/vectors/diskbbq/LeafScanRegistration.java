/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.IndexInput;
import org.elasticsearch.search.vectors.IVFKnnSearchStrategy;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

/**
 * One leaf's registration with the {@link CrossSegmentPostingScheduler}: everything the reader set up on the
 * leaf-confined thread (ranked postings, worker-state factory, collector, budget) plus the continuation that resumes
 * the reader's original serial loops on whatever the scheduler did not claim.
 *
 * <p>Thread model: the immutable fields are published to scheduler workers through the Phase A/Phase B
 * {@code invokeAll} barriers. {@link #cursor} and {@link #workerAffinity} are guarded by the scheduler's claim lock.
 * {@link #liveIterator} and {@link #postingsInput} are only ever touched by one thread at a time (the registering
 * leaf thread, then the single Phase C continuation task for this leaf).
 */
final class LeafScanRegistration {

    /** Resumes the reader's serial loops on the unclaimed remainder, under the scheduler's global visit budget. */
    interface SerialContinuation {
        void resume(CentroidIterator remaining, IVFVectorsReader.VisitBudget budget, long expectedDocs, long actualDocs) throws IOException;
    }

    final PostingMetadata[] ranked;
    final CentroidIterator liveIterator;
    final IndexInput postingsInput;
    final ParallelPostingListScanner.WorkerStateFactory workerStateFactory;
    final IVFKnnSearchStrategy strategy;
    final KnnCollector collector;
    final long maxVectorsVisited;
    final SerialContinuation continuation;

    /** Next unclaimed rank — claims are strictly a prefix per leaf; guarded by the scheduler's claim lock. */
    int cursor;
    /** Workers allowed to serve this leaf (bounds per-(worker,leaf) scoring state); guarded by the claim lock. */
    final Set<Integer> workerAffinity = new HashSet<>();
    final AtomicLong expectedDocs = new AtomicLong();
    final AtomicLong actualDocs = new AtomicLong();
    /** The leaf's final results, produced by the Phase C task for this leaf. */
    TopDocs topDocs;

    LeafScanRegistration(
        PostingMetadata[] ranked,
        CentroidIterator liveIterator,
        IndexInput postingsInput,
        ParallelPostingListScanner.WorkerStateFactory workerStateFactory,
        IVFKnnSearchStrategy strategy,
        KnnCollector collector,
        long maxVectorsVisited,
        SerialContinuation continuation
    ) {
        this.ranked = ranked;
        this.liveIterator = liveIterator;
        this.postingsInput = postingsInput;
        this.workerStateFactory = workerStateFactory;
        this.strategy = strategy;
        this.collector = collector;
        this.maxVectorsVisited = maxVectorsVisited;
        this.continuation = continuation;
    }

    /** The centroid score of the best unclaimed posting; only valid while {@code cursor < ranked.length}. */
    float headScore() {
        return ranked[cursor].documentCentroidScore();
    }
}
