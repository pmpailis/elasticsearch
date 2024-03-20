/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.search.rank.twophase;

import org.apache.lucene.search.Query;

import java.util.List;

/**
 * Executes queries and generates results on the shard for RRF.
 */
public class RandomInferenceRankShardContext extends InferenceRankShardContext {

    public RandomInferenceRankShardContext(List<Query> queries, int from, int windowSize, int rankConstant, String field, String modelId) {
        super(queries, from, windowSize, rankConstant, field, modelId);
    }

    @Override
    public float computeScore(String fieldData) {
        return (float) Math.random();
    }
}
