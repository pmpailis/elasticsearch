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
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.search.query.QuerySearchResult;
import org.elasticsearch.search.rank.rerank.RankFeatureResult;

import java.util.List;
import java.util.function.Consumer;

/**
 * {@code RankContext} is a base class used to generate ranking
 * results on the coordinator and then set the rank for any
 * search hits that are found.
 */
public abstract class RankCoordinatorContext {

    protected final int size;
    protected final int from;
    protected final int windowSize;
    protected final Client client;

    public RankCoordinatorContext(int size, int from, int windowSize, Client client) {
        this.size = size;
        this.from = from;
        this.windowSize = windowSize;
        this.client = client;
    }

    // used for faster hash lookup in a map of ranked documents
    public record RankKey(int doc, int shardIndex) {}

    public boolean isRerank() {
        return false;
    }

    /**
     * This is used to pull information passed back from the shards as part
     * of {@link QuerySearchResult#getRankShardResult()} and return a {@link SortedTopDocs}
     * of the final rank results. Note that {@link TopDocsStats} is included so that
     * appropriate stats may be updated based on rank results. This is called at the end
     * of the query phase prior to the fetch phase.
     */
    public abstract SortedTopDocs postQueryRank(List<QuerySearchResult> querySearchResults, TopDocsStats topDocStats);

    /**
     * This is used to re-rank the results after the query phase has completed. This is called
     * by the RankPhase and computes a reranked list of global top-K results, based on the features extracted
     * during this phase by each of the shards (and passed through the {@link RankFeatureResult}).
     *
     * @param rankSearchResults a list of rank feature results from each shard
     */
    public abstract void reRank(List<RankFeatureResult> rankSearchResults, Consumer<ScoreDoc[]> onFinish);
}
