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
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.IndexInput;
import org.elasticsearch.search.vectors.BulkKnnCollector;
import org.elasticsearch.search.vectors.IVFKnnSearchStrategy;
import org.elasticsearch.search.vectors.IVFParallelScanContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Scans the best-ranked posting lists of a single segment with multiple workers.
 *
 * <p>The design keeps the scoring hot path free of shared mutable state: each worker owns its {@link IndexInput}
 * slices (created on the worker's own thread), its posting visitor (and therefore its scorer, quantizer and scratch
 * buffers) and a private collector of the leaf collector's shape (so e.g. diversifying leaves get diversifying
 * workers). Coordination is limited to
 * <ul>
 *   <li>one {@link AtomicInteger} cursor over the ranked postings, advanced once per claimed chunk,</li>
 *   <li>two {@link AtomicLong} totals tracking the expected/actual visited document counts that decide when the
 *       visit budget is covered,</li>
 *   <li>a {@code LongAccumulator} through which workers publish/pull the min-competitive score once per posting
 *       list (via {@link IVFKnnSearchStrategy#nextVectorsBlock()}), mirroring the existing cross-leaf cadence.</li>
 * </ul>
 *
 * <p>Workers stop claiming chunks once the expected visited count covers the budget; postings that were drained but
 * never claimed are handed back to the caller so the serial loop can resume under its exact original termination
 * conditions. Claimed chunks are always scanned to completion, so every drained posting is either fully scored by a
 * worker or visible to the serial continuation — the visited set is a superset of the serial algorithm's.
 */
public final class ParallelPostingListScanner {

    /** Minimum drained posting lists per worker for forking to pay off. */
    static final int MIN_POSTINGS_PER_WORKER = 4;
    /** Maximum posting lists claimed per cursor advance. */
    static final int MAX_CHUNK_SIZE = 8;
    /** Extra ranked postings drained beyond the visit budget so the serial continuation rarely needs the iterator. */
    private static final double DRAIN_OVERSHOOT = 1.25;

    /**
     * Per-worker scan state. Both members must be created on the worker's own thread: the posting visitor reads
     * through its own {@link IndexInput} slice, and {@code postingsInput} is that same slice, used additionally to
     * issue chunk prefetches (prefetch does not move the file pointer).
     */
    public record WorkerState(IVFVectorsReader.PostingVisitor visitor, IndexInput postingsInput) {}

    /** Creates {@link WorkerState}; invoked on the worker thread so all inputs are thread-confined by construction. */
    public interface WorkerStateFactory {
        WorkerState create() throws IOException;
    }

    /**
     * Outcome of the parallel phase, seeding the serial continuation.
     *
     * @param expectedDocs      sum of posting sizes visited by workers (parallel counterpart of the serial loop's
     *                          {@code expectedDocs})
     * @param actualDocs        sum of documents actually scored by workers
     * @param remainingIterator unclaimed drained postings followed by the live centroid iterator, prefetch-wrapped
     */
    public record Result(long expectedDocs, long actualDocs, CentroidIterator remainingIterator) {}

    private final PostingMetadata[] ranked;
    private final AtomicInteger cursor = new AtomicInteger();
    private final AtomicLong expectedDocsTotal = new AtomicLong();
    private final AtomicLong actualDocsTotal = new AtomicLong();
    private final long maxVectorsVisited;
    private final long avgVectorsPerPosting;
    private final int workers;
    private final IVFParallelScanContext context;
    private final IVFKnnSearchStrategy leafStrategy;
    private final KnnCollector leafCollector;
    private final WorkerStateFactory workerStateFactory;

    private ParallelPostingListScanner(
        PostingMetadata[] ranked,
        long maxVectorsVisited,
        long avgVectorsPerPosting,
        int workers,
        IVFParallelScanContext context,
        IVFKnnSearchStrategy leafStrategy,
        KnnCollector leafCollector,
        WorkerStateFactory workerStateFactory
    ) {
        this.ranked = ranked;
        this.maxVectorsVisited = maxVectorsVisited;
        this.avgVectorsPerPosting = avgVectorsPerPosting;
        this.workers = workers;
        this.context = context;
        this.leafStrategy = leafStrategy;
        this.leafCollector = leafCollector;
        this.workerStateFactory = workerStateFactory;
    }

    /** Outcome of {@link #drain}: the ranked postings pulled out of the iterator and their total byte length. */
    record DrainedPostings(PostingMetadata[] ranked, long drainedBytes) {}

    /**
     * Drains the visit budget's worth (x{@link #DRAIN_OVERSHOOT}) of ranked postings out of {@code rawIterator} into
     * a plain array, prefetching the first {@code eagerPrefetch} so the earliest postings consumed are already being
     * read ahead while the rest of the drain and the worker setup happen. Must run on the thread owning both inputs.
     */
    static DrainedPostings drain(
        CentroidIterator rawIterator,
        long maxVectorsVisited,
        long approxBytesPerVector,
        int eagerPrefetch,
        IndexInput postingsInput
    ) throws IOException {
        long targetBytes = (long) (DRAIN_OVERSHOOT * maxVectorsVisited * approxBytesPerVector);
        List<PostingMetadata> drainedList = new ArrayList<>();
        long drainedBytes = 0;
        while (drainedBytes < targetBytes && rawIterator.hasNext()) {
            PostingMetadata metadata = rawIterator.nextPosting();
            if (drainedList.size() < eagerPrefetch) {
                postingsInput.prefetch(metadata.offset(), metadata.length());
            }
            drainedList.add(metadata);
            drainedBytes += metadata.length();
        }
        return new DrainedPostings(drainedList.toArray(PostingMetadata[]::new), drainedBytes);
    }

    /**
     * Drains the visit budget's worth of ranked postings from {@code rawIterator}, scans them with up to
     * {@link IVFParallelScanContext#maxWorkers()} workers and merges the worker results into {@code leafCollector}.
     * When there is too little work to fork, no worker state is created and everything drained is handed back
     * through the {@link Result} for the caller's serial loop.
     */
    public static Result scan(
        CentroidIterator rawIterator,
        long approxBytesPerVector,
        long maxVectorsVisited,
        IVFParallelScanContext context,
        IVFKnnSearchStrategy leafStrategy,
        KnnCollector leafCollector,
        WorkerStateFactory workerStateFactory,
        IndexInput callerPostingsInput
    ) throws IOException {
        DrainedPostings drained = drain(
            rawIterator,
            maxVectorsVisited,
            approxBytesPerVector,
            context.maxWorkers() * MAX_CHUNK_SIZE,
            callerPostingsInput
        );
        PostingMetadata[] ranked = drained.ranked();
        int workers = Math.min(context.maxWorkers(), ranked.length / MIN_POSTINGS_PER_WORKER);
        if (workers < 2) {
            return new Result(0, 0, prefetching(concat(ranked, 0, rawIterator), callerPostingsInput));
        }
        long avgVectorsPerPosting = Math.max(1, drained.drainedBytes() / approxBytesPerVector / ranked.length);
        ParallelPostingListScanner scanner = new ParallelPostingListScanner(
            ranked,
            maxVectorsVisited,
            avgVectorsPerPosting,
            workers,
            context,
            leafStrategy,
            leafCollector,
            workerStateFactory
        );
        scanner.run();
        int nextRank = Math.min(scanner.cursor.get(), ranked.length);
        // Lazy wrap: in the common case the budget is covered and the serial continuation never consumes the
        // remainder, so the prefetch that PrefetchingCentroidIterator issues at construction would be wasted I/O.
        CentroidIterator remaining = lazyPrefetching(concat(ranked, nextRank, rawIterator), callerPostingsInput);
        return new Result(scanner.expectedDocsTotal.get(), scanner.actualDocsTotal.get(), remaining);
    }

    private void run() throws IOException {
        List<Callable<KnnCollector>> tasks = new ArrayList<>(workers);
        for (int i = 0; i < workers; i++) {
            tasks.add(this::runWorker);
        }
        // Lucene's TaskExecutor also runs queued tasks on the calling thread, so a saturated pool degrades to
        // caller-runs instead of deadlocking, and the query thread always contributes to the scan.
        List<KnnCollector> workerCollectors = context.taskExecutor().invokeAll(tasks);
        mergeWorkerResults(workerCollectors);
    }

    private KnnCollector runWorker() throws IOException {
        WorkerState state = workerStateFactory.create();
        KnnCollector collector = leafStrategy.newParallelWorkerCollector(leafCollector.k(), leafCollector.visitLimit());
        int[] current = claimAndPrefetch(state.postingsInput());
        while (current != null) {
            context.checkCancelled().run();
            // claim (and prefetch) the next chunk before scanning the current one so the OS reads it ahead
            int[] next = claimAndPrefetch(state.postingsInput());
            score(state.visitor(), collector, current);
            current = next;
        }
        return collector;
    }

    /**
     * Claims the next chunk of ranked postings, or returns {@code null} when the visit budget is covered or the
     * ranked array is exhausted. Chunks shrink both as the array drains (so stragglers are bounded by a single
     * posting list) and as the visit budget nears (so the claimed-but-always-scanned overshoot stays small).
     */
    private int[] claimAndPrefetch(IndexInput postingsInput) throws IOException {
        long budgetRemaining = maxVectorsVisited - expectedDocsTotal.get();
        if (budgetRemaining <= 0) {
            return null;
        }
        int remaining = ranked.length - cursor.get();
        if (remaining <= 0) {
            return null;
        }
        long budgetPostings = Math.max(1, budgetRemaining / avgVectorsPerPosting);
        int size = (int) Math.max(1, Math.min(Math.min(MAX_CHUNK_SIZE, budgetPostings), remaining / (4 * workers)));
        int start = cursor.getAndAdd(size);
        if (start >= ranked.length) {
            return null;
        }
        int end = Math.min(start + size, ranked.length);
        for (int i = start; i < end; i++) {
            postingsInput.prefetch(ranked[i].offset(), ranked[i].length());
        }
        return new int[] { start, end };
    }

    private void score(IVFVectorsReader.PostingVisitor visitor, KnnCollector collector, int[] chunk) throws IOException {
        // accumulate locally and flush once per chunk so the shared totals see one RMW per chunk, not per posting
        long expected = 0;
        long actual = 0;
        for (int i = chunk[0]; i < chunk[1]; i++) {
            expected += visitor.resetPostingsScorer(ranked[i]);
            actual += visitor.visit(collector);
            collector.getSearchStrategy().nextVectorsBlock();
        }
        expectedDocsTotal.addAndGet(expected);
        actualDocsTotal.addAndGet(actual);
    }

    private void mergeWorkerResults(List<KnnCollector> workerCollectors) {
        mergeWorkerResults(workerCollectors, leafCollector);
    }

    /**
     * Drains every worker collector's private results into the leaf collector. Bulk collectors merge via a bulk heap
     * insert; others (e.g. diversifying collectors) drain through {@code topDocs()} and merge one doc at a time,
     * which re-runs the leaf collector's own dedup semantics. Every merge op is max-wins, so order cannot matter.
     */
    static void mergeWorkerResults(List<KnnCollector> workerCollectors, KnnCollector leafCollector) {
        for (KnnCollector worker : workerCollectors) {
            if (worker.visitedCount() > 0) {
                leafCollector.incVisitedCount((int) worker.visitedCount());
            }
            // non-Bulk workers (e.g. diversifying collectors) drain through topDocs(); each worker merges exactly once
            TopDocs topDocs = worker instanceof BulkKnnCollector bulkWorker ? bulkWorker.unsortedTopK() : worker.topDocs();
            int count = topDocs.scoreDocs.length;
            if (count == 0) {
                continue;
            }
            int[] docs = new int[count];
            float[] scores = new float[count];
            float bestScore = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < count; i++) {
                ScoreDoc scoreDoc = topDocs.scoreDocs[i];
                docs[i] = scoreDoc.doc;
                scores[i] = scoreDoc.score;
                if (scoreDoc.score > bestScore) {
                    bestScore = scoreDoc.score;
                }
            }
            if (leafCollector instanceof BulkKnnCollector bulkLeafCollector) {
                bulkLeafCollector.bulkCollect(docs, scores, count, bestScore);
            } else {
                for (int i = 0; i < count; i++) {
                    leafCollector.collect(docs[i], scores[i]);
                }
            }
        }
    }

    static CentroidIterator concat(PostingMetadata[] ranked, int from, CentroidIterator rest) {
        return new CentroidIterator() {
            private int next = from;

            @Override
            public boolean hasNext() {
                return next < ranked.length || rest.hasNext();
            }

            @Override
            public PostingMetadata nextPosting() throws IOException {
                return next < ranked.length ? ranked[next++] : rest.nextPosting();
            }
        };
    }

    static CentroidIterator prefetching(CentroidIterator iterator, IndexInput postingsInput) throws IOException {
        return new PrefetchingCentroidIterator(iterator, postingsInput);
    }

    /**
     * Defers the prefetch wrapping until the iterator is actually consumed: {@link PrefetchingCentroidIterator}
     * prefetches its first posting at construction, which would be wasted I/O on the common path where the serial
     * continuation never runs. {@code hasNext()} must stay side-effect free because the serial loop evaluates it
     * before its termination clauses.
     */
    static CentroidIterator lazyPrefetching(CentroidIterator iterator, IndexInput postingsInput) {
        return new CentroidIterator() {
            private CentroidIterator prefetching;

            @Override
            public boolean hasNext() {
                return prefetching != null ? prefetching.hasNext() : iterator.hasNext();
            }

            @Override
            public PostingMetadata nextPosting() throws IOException {
                if (prefetching == null) {
                    prefetching = new PrefetchingCentroidIterator(iterator, postingsInput);
                }
                return prefetching.nextPosting();
            }
        };
    }
}
