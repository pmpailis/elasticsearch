/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;

import java.io.IOException;

/**
 * A collector manager for post-filtering HNSW KNN search. Creates collectors that
 * score all documents without filter constraints and apply the filter only to the result set.
 *
 * <p>Supports Lucene's optimistic collection: in Phase 1, each leaf gets a smaller per-leaf queue
 * (proportional to the leaf's share of the index) so the HNSW beam stays small and competitive
 * thresholds tighten early. Phase 2 re-enters promising leaves with the full queue size.
 */
class PostFilteringHnswCollectorManager implements KnnCollectorManager {

    private final int queueSize;
    private final int originalK;
    private final Weight filterWeight;
    private final float selectivity;

    PostFilteringHnswCollectorManager(int queueSize, int originalK, Weight filterWeight, float selectivity) {
        this.queueSize = queueSize;
        this.originalK = originalK;
        this.filterWeight = filterWeight;
        this.selectivity = selectivity;
    }

    @Override
    public PostFilteringKnnCollector newCollector(int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context)
        throws IOException {
        return new PostFilteringKnnCollector(queueSize, originalK, visitedLimit, searchStrategy, filterWeight, context);
    }

    @Override
    public KnnCollector newOptimisticCollector(
        int visitedLimit,
        KnnSearchStrategy searchStrategy,
        LeafReaderContext context,
        int k
    ) throws IOException {
        // Scale per-leaf queue by 1/selectivity to match ACORN's exploration behavior:
        // at lower selectivity, the queue takes more visits to fill, raising the competitive
        // threshold later and causing HNSW to explore more nodes.
        int adjustedK = (int) Math.ceil(k / selectivity);
        return new PostFilteringKnnCollector(adjustedK, adjustedK, visitedLimit, searchStrategy, filterWeight, context);
    }

    @Override
    public boolean isOptimistic() {
        return true;
    }
}
