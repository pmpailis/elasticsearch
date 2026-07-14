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
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopDocs;
import org.elasticsearch.search.vectors.BulkKnnCollector;
import org.elasticsearch.search.vectors.IVFCrossSegmentRegistrar;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

import static org.elasticsearch.index.codec.vectors.diskbbq.ParallelPostingListScanner.MAX_CHUNK_SIZE;
import static org.elasticsearch.index.codec.vectors.diskbbq.ParallelPostingListScanner.MIN_POSTINGS_PER_WORKER;

/**
 * Scans the posting lists of <b>all</b> registered segments best-first, globally: workers claim the posting chunk
 * with the highest centroid score across every leaf, so the query's total visit budget flows to the most promising
 * segments instead of being split evenly. Centroid scores are cross-segment comparable — the quantization
 * corrections cancel each segment's anchor exactly, leaving only per-segment quantization noise — which is why a
 * single priority order over all leaves is meaningful.
 *
 * <p>The lifecycle has three phases, driven from the query's {@code rewrite}:
 * <ol>
 *   <li><b>Register</b> (leaf tasks, concurrent): each eligible leaf's reader performs its setup on the leaf-confined
 *       thread — centroid ranking, filter-bits materialization, budget computation, drain of the ranked postings —
 *       then registers a {@link LeafScanRegistration} and returns without scanning. Ineligible leaves complete
 *       normally inside their leaf task.</li>
 *   <li><b>Scan</b> ({@link #run()}, worker tasks): a priority heap holds each leaf's <i>next</i> chunk (so claims
 *       are strictly a per-leaf prefix, preserving the serial continuation contract). Workers claim best-first with
 *       thread-confined scoring state per (worker, leaf) — bounded by a per-leaf worker-affinity cap — and stop
 *       claiming once the <b>global</b> budget (the sum of the per-leaf budgets) is covered. In-flight chunks always
 *       finish. Phase-level reallocation is naturally capped by each leaf's drain overshoot; anything deeper flows
 *       through the continuations below.</li>
 *   <li><b>Continue</b> (per-leaf tasks): worker results merge into each leaf collector, then the reader's original
 *       serial loops resume on the unclaimed remainder — with the budget clause evaluated against the global
 *       remainder, while the per-leaf recall floors (collect-until-full and the filtered adaptive clause) run
 *       unchanged, so no leaf can be starved below the guarantees the serial algorithm provides today.</li>
 * </ol>
 *
 * <p>Correctness contract: the visited set is a superset of the <i>global best-first serial reference</i> (merge all
 * leaves' ranked streams by centroid score, visit until the global budget is covered, then run each leaf's original
 * continuation clauses). Recall can therefore differ from per-leaf-budget scanning in <i>distribution</i>, but every
 * per-leaf floor the serial algorithm guarantees is preserved.
 */
public final class CrossSegmentPostingScheduler implements IVFCrossSegmentRegistrar {

    /** Max distinct workers serving one leaf, bounding per-(worker,leaf) scoring state (quantizers, slices). */
    static final int MAX_WORKERS_PER_LEAF = 4;

    private final TaskExecutor taskExecutor;
    private final int maxWorkers;
    private final Runnable checkCancelled;

    private final Map<Integer, LeafScanRegistration> leaves = new ConcurrentHashMap<>();
    private final ReentrantLock claimLock = new ReentrantLock();
    private final PriorityQueue<LeafScanRegistration> nextChunks = new PriorityQueue<>(
        (a, b) -> Float.compare(b.headScore(), a.headScore())
    );
    private final AtomicLong globalExpected = new AtomicLong();
    private final AtomicInteger nextWorkerId = new AtomicInteger();
    private long globalBudget;

    public CrossSegmentPostingScheduler(TaskExecutor taskExecutor, int maxWorkers, Runnable checkCancelled) {
        this.taskExecutor = taskExecutor;
        this.maxWorkers = Math.max(1, maxWorkers);
        this.checkCancelled = checkCancelled;
    }

    /** Upper bound on scan workers; sizes the eager head-prefetch of each leaf's drain. */
    int maxWorkers() {
        return maxWorkers;
    }

    /** Called by the codec reader from a leaf task instead of scanning; leaf tasks register concurrently. */
    void register(int leafOrd, LeafScanRegistration registration) {
        LeafScanRegistration previous = leaves.put(leafOrd, registration);
        assert previous == null : "leaf " + leafOrd + " registered twice";
    }

    /**
     * Runs the scan and continuation phases over all registered leaves. Invoked once, on the query thread, strictly
     * after every leaf task has completed. Returns {@code false} when nothing registered (caller proceeds as today).
     */
    public boolean run() throws IOException {
        if (leaves.isEmpty()) {
            return false;
        }
        int totalRanked = 0;
        for (LeafScanRegistration leaf : leaves.values()) {
            globalBudget += leaf.maxVectorsVisited;
            if (leaf.ranked.length > 0) {
                nextChunks.add(leaf);
                totalRanked += leaf.ranked.length;
            }
        }
        List<List<LeafWorkerResult>> workerResults = List.of();
        int workers = Math.min(maxWorkers, Math.max(1, totalRanked / MIN_POSTINGS_PER_WORKER));
        if (totalRanked > 0) {
            List<Callable<List<LeafWorkerResult>>> tasks = new ArrayList<>(workers);
            for (int i = 0; i < workers; i++) {
                tasks.add(this::runWorker);
            }
            // Lucene's TaskExecutor also runs queued tasks on the calling thread, so a saturated pool degrades to
            // caller-runs instead of deadlocking, and the query thread always contributes to the scan.
            workerResults = taskExecutor.invokeAll(tasks);
        }
        // Continuation phase: one task per leaf (each leaf-confined), so continuations also run in parallel.
        Map<LeafScanRegistration, List<KnnCollector>> workerCollectorsByLeaf = new HashMap<>();
        for (List<LeafWorkerResult> perWorker : workerResults) {
            for (LeafWorkerResult result : perWorker) {
                workerCollectorsByLeaf.computeIfAbsent(result.leaf(), unused -> new ArrayList<>()).add(result.collector());
            }
        }
        List<Callable<Void>> continuations = new ArrayList<>(leaves.size());
        for (LeafScanRegistration leaf : leaves.values()) {
            List<KnnCollector> workerCollectors = workerCollectorsByLeaf.getOrDefault(leaf, List.of());
            continuations.add(() -> {
                ParallelPostingListScanner.mergeWorkerResults(workerCollectors, leaf.collector);
                CentroidIterator remaining = ParallelPostingListScanner.lazyPrefetching(
                    ParallelPostingListScanner.concat(leaf.ranked, leaf.cursor, leaf.liveIterator),
                    leaf.postingsInput
                );
                leaf.continuation.resume(remaining, globalBudget(), leaf.expectedDocs.get(), leaf.actualDocs.get());
                leaf.topDocs = leaf.collector instanceof BulkKnnCollector bulkCollector
                    ? bulkCollector.unsortedTopK()
                    : leaf.collector.topDocs();
                return null;
            });
        }
        taskExecutor.invokeAll(continuations);
        return true;
    }

    /** The final results of one registered leaf; only valid after {@link #run()}. */
    public TopDocs topDocs(int leafOrd) {
        LeafScanRegistration leaf = leaves.get(leafOrd);
        assert leaf != null && leaf.topDocs != null : "no completed registration for leaf " + leafOrd;
        return leaf.topDocs;
    }

    /** The global visit budget shared by scan workers and continuations: covered once expected docs reach it. */
    private IVFVectorsReader.VisitBudget globalBudget() {
        return new IVFVectorsReader.VisitBudget() {
            @Override
            public boolean hasRemaining() {
                return globalBudget > globalExpected.get();
            }

            @Override
            public void addExpected(long count) {
                globalExpected.addAndGet(count);
            }
        };
    }

    private record Claim(LeafScanRegistration leaf, int from, int to) {}

    private record LeafWorkerResult(LeafScanRegistration leaf, KnnCollector collector) {}

    /** A worker's thread-confined scoring state for one leaf, created lazily on first claim of that leaf. */
    private record LeafWorkerState(ParallelPostingListScanner.WorkerState workerState, KnnCollector collector) {}

    private List<LeafWorkerResult> runWorker() throws IOException {
        int workerId = nextWorkerId.getAndIncrement();
        Map<LeafScanRegistration, LeafWorkerState> states = new HashMap<>();
        Claim current = claimAndPrefetch(workerId, states);
        while (current != null) {
            checkCancelled.run();
            // claim (and prefetch) the next chunk before scoring the current one so the OS reads it ahead
            Claim next = claimAndPrefetch(workerId, states);
            score(current, states.get(current.leaf()));
            current = next;
        }
        List<LeafWorkerResult> results = new ArrayList<>(states.size());
        for (Map.Entry<LeafScanRegistration, LeafWorkerState> entry : states.entrySet()) {
            results.add(new LeafWorkerResult(entry.getKey(), entry.getValue().collector()));
        }
        return results;
    }

    private Claim claimAndPrefetch(int workerId, Map<LeafScanRegistration, LeafWorkerState> states) throws IOException {
        Claim claim = claim(workerId);
        if (claim == null) {
            return null;
        }
        LeafScanRegistration leaf = claim.leaf();
        LeafWorkerState state = states.get(leaf);
        if (state == null) {
            // everything a worker touches for this leaf is created here, on the worker's own thread
            state = new LeafWorkerState(
                leaf.workerStateFactory.create(),
                leaf.strategy.newParallelWorkerCollector(leaf.collector.k(), leaf.collector.visitLimit())
            );
            states.put(leaf, state);
        }
        for (int i = claim.from(); i < claim.to(); i++) {
            state.workerState().postingsInput().prefetch(leaf.ranked[i].offset(), leaf.ranked[i].length());
        }
        return claim;
    }

    /**
     * Claims the globally best next chunk this worker may serve, or {@code null} when the global budget is covered
     * or nothing claimable remains. The heap holds at most one entry per leaf — its <i>next</i> chunk — so per-leaf
     * claims are strictly prefix-ordered and the unclaimed suffix hands over to the continuation cleanly.
     */
    private Claim claim(int workerId) {
        claimLock.lock();
        try {
            if (globalExpected.get() >= globalBudget) {
                // budget covered: no new claims (in-flight chunks finish); continuations enforce per-leaf floors
                return null;
            }
            List<LeafScanRegistration> skipped = null;
            try {
                while (true) {
                    LeafScanRegistration leaf = nextChunks.poll();
                    if (leaf == null) {
                        return null;
                    }
                    int affinityCap = Math.min(MAX_WORKERS_PER_LEAF, maxWorkers);
                    if (leaf.workerAffinity.size() >= affinityCap && leaf.workerAffinity.contains(workerId) == false) {
                        if (skipped == null) {
                            skipped = new ArrayList<>();
                        }
                        skipped.add(leaf);
                        continue;
                    }
                    leaf.workerAffinity.add(workerId);
                    int from = leaf.cursor;
                    int to = Math.min(from + MAX_CHUNK_SIZE, leaf.ranked.length);
                    leaf.cursor = to;
                    if (to < leaf.ranked.length) {
                        nextChunks.add(leaf);
                    }
                    return new Claim(leaf, from, to);
                }
            } finally {
                if (skipped != null) {
                    nextChunks.addAll(skipped);
                }
            }
        } finally {
            claimLock.unlock();
        }
    }

    private void score(Claim claim, LeafWorkerState state) throws IOException {
        LeafScanRegistration leaf = claim.leaf();
        IVFVectorsReader.PostingVisitor visitor = state.workerState().visitor();
        KnnCollector collector = state.collector();
        // accumulate locally and flush once per chunk so the shared totals see one RMW per chunk, not per posting
        long expected = 0;
        long actual = 0;
        for (int i = claim.from(); i < claim.to(); i++) {
            expected += visitor.resetPostingsScorer(leaf.ranked[i]);
            actual += visitor.visit(collector);
            collector.getSearchStrategy().nextVectorsBlock();
        }
        leaf.expectedDocs.addAndGet(expected);
        leaf.actualDocs.addAndGet(actual);
        globalExpected.addAndGet(expected);
    }
}
