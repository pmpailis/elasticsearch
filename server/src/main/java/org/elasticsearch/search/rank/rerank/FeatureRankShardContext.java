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
import org.elasticsearch.search.SearchHits;
import org.elasticsearch.search.fetch.FetchContext;
import org.elasticsearch.search.fetch.FetchPhase;
import org.elasticsearch.search.fetch.subphase.FetchFieldsContext;
import org.elasticsearch.search.fetch.subphase.FetchFieldsPhase;
import org.elasticsearch.search.fetch.subphase.FieldAndFormat;
import org.elasticsearch.search.internal.SearchContext;
import org.elasticsearch.search.rank.RankShardContext;
import org.elasticsearch.search.rank.RankShardResult;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * {@code FeatureRankShardContext}
 */
public class FeatureRankShardContext extends RankShardContext {

    public FeatureRankShardContext(List<Query> queries, int from, int windowSize) {
        super(queries, from, windowSize);
    }

    @Override
    public RankShardResult combine(List<TopDocs> rankResults) {
        return new RankShardFeatureResult(rankResults.stream().map(topDocs -> {
            FeatureRankDoc[] featureRankDocs = new FeatureRankDoc[topDocs.scoreDocs.length];
            for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                ScoreDoc scoreDoc = topDocs.scoreDocs[i];
                featureRankDocs[i] = new FeatureRankDoc(scoreDoc.doc, scoreDoc.score, scoreDoc.shardIndex);
            }
            return featureRankDocs;
        }).flatMap(Arrays::stream).toArray(FeatureRankDoc[]::new));
//        assert rankResults.size() == 1;
//        return new RankShardFeatureResult(
//            Arrays.stream(rankResults.get(0).scoreDocs)
//                .map(x -> new FeatureRankDoc(x.doc, x.score, x.shardIndex))
//                .toArray(FeatureRankDoc[]::new));
    }

    public RankShardResult getFeatureRankDocs(SearchContext context, int[] docIds, String field) {
        // quick hack - this will definitely need to be reworked
        context.fetchFieldsContext(new FetchFieldsContext(Collections.singletonList(new FieldAndFormat(field, null))));
        FetchContext fetchContext = new FetchContext(context);
        SearchHits hits = FetchPhase.buildSearchHits(
            context,
            docIds,
            FetchPhase.Profiler.NOOP,
            Collections.singletonList(new FetchFieldsPhase().getProcessor(fetchContext))
        );
        FeatureRankDoc[] featureRankDocs = new FeatureRankDoc[hits.getHits().length];
        for (int i = 0; i < hits.getHits().length; i++) {
            featureRankDocs[i] = new FeatureRankDoc(docIds[i], hits.getHits()[i].getScore(), context.shardTarget().getShardId().id());
            featureRankDocs[i].featureData(hits.getHits()[i].field(field).getValue());
        }
        hits.decRef();
        return new RankShardFeatureResult(featureRankDocs);
    }
}
