/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Per-leaf producer of {@link PostingMetadata} for the parallel IVF search path.
 *
 * <p>Wraps a {@link CentroidIterator} together with the {@link IVFVectorsReader.PostingVisitor} that will score
 * its posting lists and the {@link IndexInput} slice used to issue prefetch hints. Each call to
 * {@link #yieldNext()} advances the underlying centroid iterator and asks the OS to bring the
 * next posting list's bytes into cache so the worker that picks up the task does not pay the I/O
 * cost itself.
 *
 * <p>The visitor also drives an adaptive batch schedule. The initial batch — supplied at
 * construction — is roughly {@code visitRatio * numCentroids / numLeaves}, i.e. how many posting
 * lists this leaf is expected to score in total, evenly amortized across the cluster. Each
 * subsequent batch returned by {@link #nextBatch()} is half the previous one (floor 1) so we
 * front-load prefetch aggressively while the queue is being seeded but taper down once the work
 * is in flight, avoiding wasted reads on centroids that may be skipped by early termination.
 *
 * <p>This class is thread-safe: {@link #yieldNext()} is synchronized so concurrent refills from
 * different worker threads do not race on the iterator. Posting scoring itself is serialized
 * separately by the per-leaf scorer that consumes these tasks.
 */
public final class BatchedPostingVisitor {

    private final CentroidIterator centroidIterator;
    private final IVFVectorsReader.PostingVisitor postingVisitor;
    private final IndexInput postingListSlice;
    private final int numVectors;
    private final long maxVectorVisited;
    private final boolean filtered;
    private final float percentFiltered;
    private final AtomicInteger inFlight = new AtomicInteger(0);

    private int nextBatchSize;

    public BatchedPostingVisitor(
        CentroidIterator centroidIterator,
        IVFVectorsReader.PostingVisitor postingVisitor,
        IndexInput postingListSlice,
        int initialBatchSize,
        int numVectors,
        long maxVectorVisited,
        boolean filtered,
        float percentFiltered
    ) {
        if (initialBatchSize < 1) {
            throw new IllegalArgumentException("initialBatchSize must be at least 1, got: " + initialBatchSize);
        }
        this.centroidIterator = centroidIterator;
        this.postingVisitor = postingVisitor;
        this.postingListSlice = postingListSlice;
        this.numVectors = numVectors;
        this.maxVectorVisited = maxVectorVisited;
        this.filtered = filtered;
        this.percentFiltered = percentFiltered;
        this.nextBatchSize = initialBatchSize;
    }

    public IVFVectorsReader.PostingVisitor postingVisitor() {
        return postingVisitor;
    }

    public int numVectors() {
        return numVectors;
    }

    public long maxVectorVisited() {
        return maxVectorVisited;
    }

    public boolean filtered() {
        return filtered;
    }

    public float percentFiltered() {
        return percentFiltered;
    }

    public synchronized boolean hasNext() {
        return centroidIterator.hasNext();
    }

    /**
     * Advances the centroid iterator by one and issues a prefetch for the returned posting list's
     * bytes. Increments the in-flight counter. Returns {@code null} when the iterator is exhausted.
     */
    public synchronized PostingMetadata yieldNext() throws IOException {
        if (centroidIterator.hasNext() == false) {
            return null;
        }
        PostingMetadata md = centroidIterator.nextPosting();
        // Hint the OS to start loading bytes for this posting list. The actual read happens later
        // on the worker that pulls this task off the queue; the prefetch overlaps with whatever
        // scoring is currently in flight.
        postingListSlice.prefetch(md.offset(), md.length());
        inFlight.incrementAndGet();
        return md;
    }

    /**
     * Returns the size of the next batch of tasks the orchestrator should seed from this leaf.
     * Each call halves the value (floor 1), implementing the adaptive halving schedule.
     */
    public synchronized int nextBatch() {
        int batch = nextBatchSize;
        nextBatchSize = Math.max(1, nextBatchSize / 2);
        return batch;
    }

    /**
     * Decrements the in-flight counter after a task yielded by this visitor has been fully
     * processed. Returns the remaining number of tasks yielded but not yet completed; when this
     * reaches zero the orchestrator should kick off the next (halved) batch.
     */
    public int onTaskCompleted() {
        return inFlight.decrementAndGet();
    }
}
