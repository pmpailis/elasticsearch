/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.search.rank.twophase;

import org.apache.lucene.search.Query;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.SearchHits;
import org.elasticsearch.search.fetch.FetchContext;
import org.elasticsearch.search.fetch.FetchPhase;
import org.elasticsearch.search.fetch.subphase.FetchFieldsContext;
import org.elasticsearch.search.fetch.subphase.FetchFieldsPhase;
import org.elasticsearch.search.fetch.subphase.FieldAndFormat;
import org.elasticsearch.search.internal.SearchContext;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Executes queries and generates results on the shard for RRF.
 */
public abstract class InferenceRankShardContext extends TwoPhaseRankShardContext {

    private final String field;
    private final String modelId;

    public InferenceRankShardContext(List<Query> queries, int from, int windowSize, int rankConstant, String field, String modelId) {
        super(queries, from, windowSize, rankConstant);
        this.field = field;
        this.modelId = modelId;
    }

    public TwoPhaseRankShardResult computeUpdatedScores(SearchContext context, int[] docs, float[] scores) {
        TwoPhaseRankDoc[] rankDocs = new TwoPhaseRankDoc[docs.length];
        int processed = 0;
        try {
            Arrays.sort(docs);
            context.fetchFieldsContext(new FetchFieldsContext(List.of(new FieldAndFormat(field, null))));
            FetchContext fetchContext = new FetchContext(context);
            SearchHits hits = FetchPhase.buildSearchHits(
                context,
                docs,
                FetchPhase.Profiler.NOOP,
                Collections.singletonList(new FetchFieldsPhase().getProcessor(fetchContext))
            );
            for (SearchHit hit : hits) {
                float score = computeScore(hit.getFields().get(field).getValue());
                rankDocs[processed++] = new TwoPhaseRankDoc(hit.docId(), hit.getScore(), context.shardTarget().getShardId().id(), score);
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
        Arrays.sort(rankDocs, Comparator.comparing(TwoPhaseRankDoc::getSecondPhaseScore).reversed());
        return new TwoPhaseRankShardResult(rankDocs);
    }

    abstract float computeScore(String fieldData);
}
