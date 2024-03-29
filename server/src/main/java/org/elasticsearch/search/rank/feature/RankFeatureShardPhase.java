/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank.feature;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.search.internal.SearchContext;
import org.elasticsearch.search.rank.RankFeaturePhaseShardContext;

import java.util.Arrays;

public class RankFeatureShardPhase {
    private static final Logger LOGGER = LogManager.getLogger(RankFeatureShardPhase.class);

    public static void execute(SearchContext searchContext, RankFeatureShardRequest request) {
        RankFeaturePhaseShardContext rankFeaturePhaseShardContext = searchContext.request().source().rankBuilder() != null
            ? searchContext.request().source().rankBuilder().buildFeaturePhaseShardContext(searchContext)
            : null;
        if (rankFeaturePhaseShardContext != null) {
            String field = request.getFeatureField();
            int[] docIds = request.getDocIds();
            Arrays.sort(docIds);
            RankFeatureShardResult featureRankShardResult = (RankFeatureShardResult) rankFeaturePhaseShardContext.fetchFeatureData(
                docIds,
                field
            );
            searchContext.rankFeatureResult().shardResult(featureRankShardResult);
        }
    }
}
