/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank.rerank;

import org.apache.lucene.search.ScoreDoc;
import org.elasticsearch.action.search.SearchPhaseController;
import org.elasticsearch.search.query.QuerySearchResult;
import org.elasticsearch.search.rank.QueryPhaseCoordinatorContext;
import org.elasticsearch.search.rank.feature.RankFeatureDoc;
import org.elasticsearch.search.rank.feature.RankFeatureShardResult;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class RerankingQueryPhaseCoordinatorContext extends QueryPhaseCoordinatorContext {

    private final int from;

    public RerankingQueryPhaseCoordinatorContext(int windowSize, int from) {
        super(windowSize);
        this.from = from;
    }

    @Override
    public ScoreDoc[] rankQueryPhaseResults(List<QuerySearchResult> querySearchResults, SearchPhaseController.TopDocsStats topDocStats) {
        List<RankFeatureDoc> rankDocs = new ArrayList<>();
        for (int i = 0; i < querySearchResults.size(); i++) {
            QuerySearchResult querySearchResult = querySearchResults.get(i);
            RankFeatureShardResult shardResult = (RankFeatureShardResult) querySearchResult.getRankShardResult();
            for (RankFeatureDoc frd : shardResult.rankFeatureDocs) {
                frd.shardIndex = i;
                rankDocs.add(frd);
            }
        }
        // no support for sort field atm
        // should pass needed info to make use of org.elasticsearch.action.search.SearchPhaseController.sortDocs?
        rankDocs.sort(Comparator.comparing((RankFeatureDoc doc) -> doc.score).reversed());
        RankFeatureDoc[] topResults = rankDocs.stream().limit(windowSize).toArray(RankFeatureDoc[]::new);

        assert topDocStats.fetchHits == 0;
        topDocStats.fetchHits = topResults.length;

        return topResults;
    }
}
