/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank.rerank;

import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.elasticsearch.search.rank.QueryPhaseShardContext;
import org.elasticsearch.search.rank.RankShardResult;
import org.elasticsearch.search.rank.feature.RankFeatureDoc;
import org.elasticsearch.search.rank.feature.RankFeatureShardResult;

import java.util.Arrays;
import java.util.List;

/**
 * {@code FeatureRankShardContext}
 */
public class RerankingQueryPhaseShardContext extends QueryPhaseShardContext {

    public RerankingQueryPhaseShardContext(List<Query> queries, int windowSize) {
        super(queries, windowSize);
    }

    @Override
    public RankShardResult combineQueryPhaseResults(List<TopDocs> rankResults) {
        RankFeatureDoc[] rankDocs = rankResults.stream().map(topDocs -> {
            RankFeatureDoc[] rankFeatureDocs = new RankFeatureDoc[topDocs.scoreDocs.length];
            for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                ScoreDoc scoreDoc = topDocs.scoreDocs[i];
                rankFeatureDocs[i] = new RankFeatureDoc(scoreDoc.doc, scoreDoc.score, scoreDoc.shardIndex);
            }
            return rankFeatureDocs;
        }).flatMap(Arrays::stream).toArray(RankFeatureDoc[]::new);
        Arrays.sort(rankDocs, (o1, o2) -> Float.compare(o2.score, o1.score));
        return new RankFeatureShardResult(rankDocs);
    }
}
