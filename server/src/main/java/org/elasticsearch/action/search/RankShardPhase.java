/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */
package org.elasticsearch.action.search;

import org.elasticsearch.search.internal.SearchContext;
import org.elasticsearch.search.rank.RankShardContext;
import org.elasticsearch.search.rank.twophase.TwoPhaseRankShardContext;
import org.elasticsearch.search.rank.twophase.TwoPhaseRankShardResult;

/**
* ranksearchphase
 */
public final class RankShardPhase {

    public static void execute(SearchContext searchContext, int[] docIds) {
        if (searchContext.rankShardContext() != null) {
            RankShardContext rankShardContext = searchContext.rankShardContext();
            if (rankShardContext instanceof TwoPhaseRankShardContext) {
                TwoPhaseRankShardResult updatedRanks = ((TwoPhaseRankShardContext) rankShardContext).computeUpdatedScores(
                    searchContext,
                    docIds,
                    new float[0]
                );
                searchContext.rankSearchResult().shardResult(updatedRanks);
            }
        }
    }
}
