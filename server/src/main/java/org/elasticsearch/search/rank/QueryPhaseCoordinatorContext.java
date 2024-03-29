/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank;

import org.apache.lucene.search.ScoreDoc;
import org.elasticsearch.action.search.SearchPhaseController.SortedTopDocs;
import org.elasticsearch.action.search.SearchPhaseController.TopDocsStats;
import org.elasticsearch.search.query.QuerySearchResult;

import java.util.List;

/**
 * {@code PostQueryCoordinatorContext} is a base class used to generate ranking
 * results on the coordinator and then set the rank for any
 * search hits that are found.
 */
public abstract class QueryPhaseCoordinatorContext {

    protected final int windowSize;

    public QueryPhaseCoordinatorContext(int windowSize) {
        this.windowSize = windowSize;
    }

    /**
     * This is used to pull information passed back from the shards as part
     * of {@link QuerySearchResult#getRankShardResult()} and return a {@link SortedTopDocs}
     * of the `window_size` ranked results. Note that {@link TopDocsStats} is included so that
     * appropriate stats may be updated based on rank results. This is called at the end
     * of the query phase prior to the fetch phase.
     */
    public abstract ScoreDoc[] rankQueryPhaseResults(List<QuerySearchResult> querySearchResults, TopDocsStats topDocStats);

}
