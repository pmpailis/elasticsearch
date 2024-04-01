/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank.rerank;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.search.SearchHits;
import org.elasticsearch.search.fetch.FetchContext;
import org.elasticsearch.search.fetch.FetchPhase;
import org.elasticsearch.search.fetch.subphase.FetchFieldsContext;
import org.elasticsearch.search.fetch.subphase.FetchFieldsPhase;
import org.elasticsearch.search.fetch.subphase.FieldAndFormat;
import org.elasticsearch.search.internal.SearchContext;
import org.elasticsearch.search.rank.RankFeaturePhaseRankShardContext;
import org.elasticsearch.search.rank.RankShardResult;
import org.elasticsearch.search.rank.feature.RankFeatureDoc;
import org.elasticsearch.search.rank.feature.RankFeatureShardResult;

import java.util.Arrays;
import java.util.Collections;

/**
 * {@link RerankingRankFeaturePhaseRankShardContext} is running on each shard is responsible for extracting string data for a set of docids
 * for a given field, and pass them back as {@link RankFeatureShardResult}.
 */
public class RerankingRankFeaturePhaseRankShardContext extends RankFeaturePhaseRankShardContext {

    private static final Logger LOGGER = LogManager.getLogger(RerankingRankFeaturePhaseRankShardContext.class);

    private final String field;
    private final SearchContext context;

    public RerankingRankFeaturePhaseRankShardContext(SearchContext context, String field) {
        super();
        this.context = context;
        this.field = field;
    }

    // This currently makes use of a new FetchContext initialized just with the FetchFieldsPhase processor, so that we build
    // search hits containing info on just the requested field. This should probably need to be revisited and maybe reworked.
    @Nullable
    public RankShardResult fetchFeatureData(int[] docIds) {
        context.fetchFieldsContext(new FetchFieldsContext(Collections.singletonList(new FieldAndFormat(field, null))));
        FetchContext fetchContext = new FetchContext(context);
        SearchHits hits = null;
        try {
            hits = FetchPhase.buildSearchHits(
                context,
                docIds,
                FetchPhase.Profiler.NOOP,
                Collections.singletonList(new FetchFieldsPhase().getProcessor(fetchContext))
            );

            RankFeatureDoc[] rankFeatureDocs = new RankFeatureDoc[hits.getHits().length];
            for (int i = 0; i < hits.getHits().length; i++) {
                rankFeatureDocs[i] = new RankFeatureDoc(docIds[i], hits.getHits()[i].getScore(), context.shardTarget().getShardId().id());
                rankFeatureDocs[i].featureData(hits.getHits()[i].field(field).getValue());
            }
            return new RankFeatureShardResult(rankFeatureDocs);
        } catch (Exception ex) {
            LOGGER.info("Error while fetching feature data for {field: " + field + "} and {docids: " + Arrays.toString(docIds) + "}.", ex);
            return null;
        } finally {
            if (hits != null) {
                hits.decRef();
            }
        }
    }
}
