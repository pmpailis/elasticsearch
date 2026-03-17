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
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;

import java.io.IOException;

/**
 * A collector manager for post-filtering HNSW KNN search. Creates collectors that
 * score all documents without filter constraints and apply the filter only to the result set.
 */
class PostFilteringHnswCollectorManager implements KnnCollectorManager {

    private final int oversampledK;
    private final int originalK;
    private final Weight filterWeight;

    PostFilteringHnswCollectorManager(int oversampledK, int originalK, Weight filterWeight) {
        this.oversampledK = oversampledK;
        this.originalK = originalK;
        this.filterWeight = filterWeight;
    }

    @Override
    public PostFilteringKnnCollector newCollector(int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context)
        throws IOException {
        return new PostFilteringKnnCollector(oversampledK, originalK, visitedLimit, searchStrategy, filterWeight, context);
    }
}
