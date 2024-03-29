/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank;

import org.apache.lucene.search.ScoreDoc;
import org.elasticsearch.action.search.RankFeaturePhase;
import org.elasticsearch.search.rank.feature.RankFeatureResult;

import java.util.List;
import java.util.function.Consumer;

/**
 * {@code RankContext} is a base class used to generate ranking
 * results on the coordinator and then set the rank for any
 * search hits that are found.
 */
public abstract class RankFeaturePhaseCoordinatorContext {

    protected final int size;
    protected final int from;
    protected final int windowSize;

    public RankFeaturePhaseCoordinatorContext(int size, int from, int windowSize) {
        this.size = size;
        this.from = from;
        this.windowSize = windowSize;
    }

    /**
     * This is used to re-rank the results after the query phase has completed. This is called
     * by the {@link RankFeaturePhase} and computes a reranked list of global top-K results, based on the features extracted
     * during this phase by each of the shards (and passed through the {@link RankFeatureResult}).
     *
     * @param rankSearchResults a list of rank feature results from each shard
     */
    public abstract void rankGlobalResults(List<RankFeatureResult> rankSearchResults, Consumer<ScoreDoc[]> onFinish);
}
