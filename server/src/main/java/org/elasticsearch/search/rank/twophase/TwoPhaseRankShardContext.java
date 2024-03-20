/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.search.rank.twophase;

import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.elasticsearch.search.internal.SearchContext;
import org.elasticsearch.search.rank.RankShardContext;

import java.util.List;

/**
 * Executes queries and generates results on the shard for RRF.
 */
public abstract class TwoPhaseRankShardContext extends RankShardContext {

    protected final int rankConstant;

    public TwoPhaseRankShardContext(List<Query> queries, int from, int windowSize, int rankConstant) {
        super(queries, from, windowSize);
        this.rankConstant = rankConstant;
    }

    @Override
    public TwoPhaseRankShardResult combine(List<TopDocs> rankResults) {
        // just sort the docs (if there are more than one - need to check this for the retriever scenario)
        TopDocs topDocs = rankResults.get(0);
        TwoPhaseRankDoc[] results = new TwoPhaseRankDoc[topDocs.scoreDocs.length];
        for (int i = 0; i < results.length; i++) {
            results[i] = new TwoPhaseRankDoc(topDocs.scoreDocs[i].doc, topDocs.scoreDocs[i].score, topDocs.scoreDocs[i].shardIndex, 0.0f);
        }
        return new TwoPhaseRankShardResult(results);
    }

    public abstract TwoPhaseRankShardResult computeUpdatedScores(SearchContext context, int[] docids, float[] scores);
}
