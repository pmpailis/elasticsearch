/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import org.elasticsearch.core.Nullable;

import java.util.concurrent.atomic.LongAccumulator;

import static org.elasticsearch.search.vectors.AbstractMaxScoreKnnCollector.LEAST_COMPETITIVE;

/**
 * Min-competitive score floors shared between the collectors of one IVF search.
 *
 * <p>The {@code local} accumulator couples the collectors of a single leaf — the leaf collector and its
 * intra-segment parallel workers — so they prune against each other's results. The {@code global} accumulator, when
 * present, additionally couples all leaves of the query: a segment holding no globally competitive documents stops
 * collecting (its blocks get pruned) as soon as other segments have produced better results.
 *
 * <p>Publishing to the global floor is gated per leaf. A floor published at heap rank {@code k_leaf} certifies that
 * many heap <i>entries</i>, not distinct documents: SOAR stores a vector in up to two posting lists and the
 * duplicates are only removed after collection, so a plain top-k heap may hold two entries per doc. A floor is
 * globally valid iff it certifies at least {@code mergeK} (the shard's final cut) distinct documents:
 * {@code k_leaf >= 2 * mergeK} for plain collectors, {@code k_leaf >= mergeK} for diversifying collectors (their
 * heap entries are distinct parents). Consuming a valid global floor is always safe — a doc below it cannot make the
 * merged top-{@code mergeK} — so {@link #floor()} folds it in unconditionally.
 */
final class ScoreFloors {

    private final LongAccumulator local;
    @Nullable
    private final LongAccumulator global;
    private final boolean publishGlobal;

    ScoreFloors(@Nullable LongAccumulator global, boolean publishGlobal) {
        this.local = new LongAccumulator(Long::max, LEAST_COMPETITIVE);
        this.global = global;
        this.publishGlobal = publishGlobal;
    }

    /**
     * Publishes a collector's floor (already guarded: only full heaps export their top, see
     * {@link AbstractMaxScoreKnnCollector#getMinCompetitiveDocScore}).
     */
    void publish(long floor) {
        local.accumulate(floor);
        if (publishGlobal && global != null) {
            global.accumulate(floor);
        }
    }

    /** The best floor visible to this leaf's collectors: the leaf-local one folded with the global one, if any. */
    long floor() {
        long floor = local.get();
        return global == null ? floor : Math.max(floor, global.get());
    }
}
