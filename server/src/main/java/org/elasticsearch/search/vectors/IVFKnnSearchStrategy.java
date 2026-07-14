/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.util.SetOnce;

import java.util.Objects;

public class IVFKnnSearchStrategy extends KnnSearchStrategy {
    private final float visitRatio;
    private final int numCands;
    private final int k;
    private final SetOnce<AbstractMaxScoreKnnCollector> collector = new SetOnce<>();
    private final ScoreFloors floors;
    private final IVFParallelScanContext parallelScanContext;
    private final IVFCrossSegmentRegistrar crossSegmentRegistrar;
    private final int crossSegmentLeafOrd;
    private boolean crossSegmentRegistered;

    public IVFKnnSearchStrategy(float visitRatio, int numCands, int k, ScoreFloors floors) {
        this(visitRatio, numCands, k, floors, null);
    }

    public IVFKnnSearchStrategy(float visitRatio, int numCands, int k, ScoreFloors floors, IVFParallelScanContext parallelScanContext) {
        this(visitRatio, numCands, k, floors, parallelScanContext, null, -1);
    }

    public IVFKnnSearchStrategy(
        float visitRatio,
        int numCands,
        int k,
        ScoreFloors floors,
        IVFParallelScanContext parallelScanContext,
        IVFCrossSegmentRegistrar crossSegmentRegistrar,
        int crossSegmentLeafOrd
    ) {
        this.visitRatio = visitRatio;
        this.numCands = numCands;
        this.k = k;
        this.floors = floors;
        this.parallelScanContext = parallelScanContext;
        this.crossSegmentRegistrar = crossSegmentRegistrar;
        this.crossSegmentLeafOrd = crossSegmentLeafOrd;
    }

    /**
     * Context for within-segment parallel scanning of posting lists, or {@code null} when this leaf must be scanned
     * serially.
     */
    public IVFParallelScanContext parallelScanContext() {
        return parallelScanContext;
    }

    /**
     * The query-level cross-segment scheduler this leaf should register with instead of scanning, or {@code null}.
     */
    public IVFCrossSegmentRegistrar crossSegmentRegistrar() {
        return crossSegmentRegistrar;
    }

    /** The ordinal of this leaf within the query's reader, valid when {@link #crossSegmentRegistrar()} is non-null. */
    public int crossSegmentLeafOrd() {
        return crossSegmentLeafOrd;
    }

    /** Set by the codec reader when it registered the leaf with the cross-segment scheduler instead of scanning. */
    public void markCrossSegmentRegistered() {
        this.crossSegmentRegistered = true;
    }

    /** Only meaningful on the thread that ran the search — the reader marks it before returning. */
    public boolean crossSegmentRegistered() {
        return crossSegmentRegistered;
    }

    /**
     * Creates a private collector of the leaf collector's shape for one parallel worker, so workers preserve the
     * leaf collector's semantics (e.g. per-parent diversification). The worker's strategy shares this leaf's
     * {@link ScoreFloors} so workers prune against each other's results (and, under cross-leaf sharing, against the
     * other leaves') without any shared mutable collector state on the scoring hot path. Must only be called after
     * {@link #setCollector}; safe to call on the worker's own thread.
     */
    public KnnCollector newParallelWorkerCollector(int collectorK, long visitLimit) {
        ScoreFloors workerFloors = parallelScanContext != null ? parallelScanContext.floors() : floors;
        assert workerFloors != null : "parallel worker collectors require shared score floors";
        AbstractMaxScoreKnnCollector leafCollector = collector.get();
        assert leafCollector != null : "the leaf collector must be set before creating worker collectors";
        IVFKnnSearchStrategy workerStrategy = new IVFKnnSearchStrategy(visitRatio, numCands, k, workerFloors);
        AbstractMaxScoreKnnCollector workerCollector = leafCollector.newParallelWorkerCollector(collectorK, visitLimit, workerStrategy);
        workerStrategy.setCollector(workerCollector);
        return workerCollector;
    }

    void setCollector(AbstractMaxScoreKnnCollector collector) {
        this.collector.set(collector);
        if (floors != null) {
            collector.updateMinCompetitiveDocScore(floors.floor());
        }
    }

    public float getVisitRatio() {
        return visitRatio;
    }

    public int getNumCands() {
        return numCands;
    }

    public int getK() {
        return k;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        IVFKnnSearchStrategy that = (IVFKnnSearchStrategy) o;
        return visitRatio == that.visitRatio && numCands == that.numCands && k == that.k;
    }

    @Override
    public int hashCode() {
        return Objects.hash(visitRatio, numCands, k);
    }

    /**
     * Called once per posting list: publishes the collector's guarded min-competitive floor to the shared
     * {@link ScoreFloors} (leaf-local always; query-global when this leaf's publish gate allows it) and pulls the
     * best visible floor back into the collector so it prunes against the other collectors' results.
     */
    @Override
    public void nextVectorsBlock() {
        if (floors == null) {
            return;
        }
        assert this.collector.get() != null : "Collector must be set before nextVectorsBlock is called";
        AbstractMaxScoreKnnCollector knnCollector = this.collector.get();
        long collectorScore = knnCollector.getMinCompetitiveDocScore();
        floors.publish(collectorScore);
        long currentScore = floors.floor();
        if (currentScore > collectorScore) {
            knnCollector.updateMinCompetitiveDocScore(currentScore);
        }
    }
}
