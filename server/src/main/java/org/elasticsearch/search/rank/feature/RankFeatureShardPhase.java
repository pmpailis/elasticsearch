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
import org.elasticsearch.search.SearchContextSourcePrinter;
import org.elasticsearch.search.internal.SearchContext;
import org.elasticsearch.search.rank.RankFeaturePhaseRankShardContext;
import org.elasticsearch.tasks.TaskCancelledException;

import java.util.Arrays;
import java.util.Objects;

public class RankFeatureShardPhase {

    private static final Logger LOGGER = LogManager.getLogger(RankFeatureShardPhase.class);

    private static final RankFeatureShardResult EMPTY_RESULT = new RankFeatureShardResult(new RankFeatureDoc[0]);

    public static void execute(SearchContext searchContext, RankFeatureShardRequest request) {
        if (LOGGER.isTraceEnabled()) {
            LOGGER.trace("{}", new SearchContextSourcePrinter(searchContext));
        }

        if (searchContext.isCancelled()) {
            throw new TaskCancelledException("cancelled");
        }

        RankFeaturePhaseRankShardContext rankFeaturePhaseRankShardContext = searchContext.request().source().rankBuilder() != null
            ? searchContext.request().source().rankBuilder().buildRankFeaturePhaseShardContext(searchContext)
            : null;
        if (rankFeaturePhaseRankShardContext != null) {
            int[] docIds = request.getDocIds();
            if (docIds == null || docIds.length == 0) {
                // no individual hits to process, so we shortcut
                searchContext.rankFeatureResult().shardResult(EMPTY_RESULT);
                return;
            }
            // ensure that docids are sorted
            Arrays.sort(docIds);
            RankFeatureShardResult featureRankShardResult = null;
            try {
                featureRankShardResult = (RankFeatureShardResult) rankFeaturePhaseRankShardContext.fetchFeatureData(docIds);
            } finally {
                // save the result in the search context
                searchContext.rankFeatureResult().shardResult(Objects.requireNonNullElse(featureRankShardResult, EMPTY_RESULT));
            }
        }
    }
}
