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
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnSearchStrategy;

import java.io.IOException;

/**
 * A collector manager for post-filtering IVF KNN search. Instead of passing the filter
 * as AcceptDocs to the codec (which can hurt SIMD bulk scoring), this creates collectors
 * that score all documents unfiltered and apply the filter only to the result set.
 */
class PostFilteringIVFCollectorManager extends AbstractIVFKnnVectorQuery.IVFCollectorManager {

    private final int oversampledK;
    private final int originalK;
    private final Weight filterWeight;

    PostFilteringIVFCollectorManager(int oversampledK, int originalK, IndexSearcher searcher, Weight filterWeight) {
        super(oversampledK, searcher);
        this.oversampledK = oversampledK;
        this.originalK = originalK;
        this.filterWeight = filterWeight;
    }

    @Override
    public AbstractMaxScoreKnnCollector newCollector(int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context)
        throws IOException {
        ScorerSupplier ss = filterWeight.scorerSupplier(context);
        if (ss == null) {
            return null;
        }
        return new PostFilteringKnnCollector(oversampledK, originalK, visitedLimit, searchStrategy, filterWeight, context);
    }
}
