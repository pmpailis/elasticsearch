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
import org.elasticsearch.search.SearchHits;
import org.elasticsearch.search.fetch.FetchPhase;
import org.elasticsearch.search.fetch.FetchSubPhase;
import org.elasticsearch.search.fetch.subphase.FetchFieldsContext;
import org.elasticsearch.search.fetch.subphase.FetchFieldsPhase;
import org.elasticsearch.search.fetch.subphase.FieldAndFormat;
import org.elasticsearch.search.internal.SearchContext;
import org.elasticsearch.search.rank.RankFeaturePhaseRankShardContext;
import org.elasticsearch.tasks.TaskCancelledException;

import java.util.Arrays;
import java.util.Collections;
import java.util.Objects;

public class RankFeatureShardPhase extends FetchPhase {

    private static final Logger LOGGER = LogManager.getLogger(RankFeatureShardPhase.class);

    private static final FetchSubPhase fetchFieldsPhase = new FetchFieldsPhase();

    private static final RankFeatureShardResult EMPTY_RESULT = new RankFeatureShardResult(new RankFeatureDoc[0]);

    public RankFeatureShardPhase() {
        super(Collections.singletonList(fetchFieldsPhase));
    }

    public void execute(SearchContext searchContext, RankFeatureShardRequest request) {
        if (LOGGER.isTraceEnabled()) {
            LOGGER.trace("{}", new SearchContextSourcePrinter(searchContext));
        }

        if (searchContext.isCancelled()) {
            throw new TaskCancelledException("cancelled");
        }

        RankFeaturePhaseRankShardContext rankFeaturePhaseRankShardContext = searchContext.request().source().rankBuilder() != null
            ? searchContext.request().source().rankBuilder().buildRankFeaturePhaseShardContext()
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
            SearchHits hits = null;
            try {
                searchContext.fetchFieldsContext(
                    new FetchFieldsContext(Collections.singletonList(new FieldAndFormat(rankFeaturePhaseRankShardContext.getField(), null)))
                );
                hits = buildSearchHits(searchContext, docIds, FetchPhase.Profiler.NOOP);
                featureRankShardResult = (RankFeatureShardResult) rankFeaturePhaseRankShardContext.buildRankFeatureShardResult(
                    hits,
                    searchContext.shardTarget().getShardId().id()
                );
            } finally {
                if (hits != null) {
                    hits.decRef();
                }
                // save the result in the search context
                searchContext.rankFeatureResult().shardResult(Objects.requireNonNullElse(featureRankShardResult, EMPTY_RESULT));
            }
        }
    }
}
