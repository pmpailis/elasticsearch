/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank.rerank;

import org.elasticsearch.search.SearchHits;
import org.elasticsearch.search.fetch.FetchContext;
import org.elasticsearch.search.fetch.FetchPhase;
import org.elasticsearch.search.fetch.subphase.FetchFieldsContext;
import org.elasticsearch.search.fetch.subphase.FetchFieldsPhase;
import org.elasticsearch.search.fetch.subphase.FieldAndFormat;
import org.elasticsearch.search.internal.SearchContext;
import org.elasticsearch.search.rank.RankFeaturePhaseShardContext;
import org.elasticsearch.search.rank.RankShardResult;
import org.elasticsearch.search.rank.feature.RankFeatureDoc;
import org.elasticsearch.search.rank.feature.RankFeatureShardResult;

import java.util.Collections;

/**
 * {@code FeatureRankShardContext}
 */
public class RerankingRankFeaturePhaseShardContext extends RankFeaturePhaseShardContext {

    private final SearchContext context;

    public RerankingRankFeaturePhaseShardContext(SearchContext context) {
        super();
        this.context = context;
    }

    public RankShardResult fetchFeatureData(int[] docIds, String field) {
        // quick hack - this will definitely need to be reworked
        context.fetchFieldsContext(new FetchFieldsContext(Collections.singletonList(new FieldAndFormat(field, null))));
        FetchContext fetchContext = new FetchContext(context);
        SearchHits hits = FetchPhase.buildSearchHits(
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
        hits.decRef();
        return new RankFeatureShardResult(rankFeatureDocs);
    }
}
