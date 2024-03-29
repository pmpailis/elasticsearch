/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.core.ml.reranking;

import org.elasticsearch.action.ActionType;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.inference.InferenceServiceResults;
import org.elasticsearch.inference.InputType;
import org.elasticsearch.inference.TaskType;
import org.elasticsearch.xpack.core.inference.action.InferenceAction;
import org.elasticsearch.xpack.core.ml.inference.results.TextEmbeddingResults;

import java.util.List;
import java.util.Map;

public class CrossEncoderRankFeaturePhaseCoordinatorContext extends InferenceRankFeaturePhaseCoordinatorContext<
    InferenceAction.Request,
    InferenceAction.Response> {

    public CrossEncoderRankFeaturePhaseCoordinatorContext(int size, int from, int windowSize, Client client) {
        super(size, from, windowSize, client);
    }

    @Override
    protected InferenceAction.Request request(List<String> docFeatures) {
        return new InferenceAction.Request(TaskType.TEXT_EMBEDDING, inferenceId, docFeatures, Map.of(), InputType.SEARCH);
    }

    @Override
    protected ActionType<InferenceAction.Response> action() {
        return InferenceAction.INSTANCE;
    }

    @Override
    protected double[] extractScoresFromResponse(InferenceAction.Response response) {
        InferenceServiceResults results = response.getResults();
        assert results instanceof TextEmbeddingResults;
        return ((TextEmbeddingResults) results).getInference();
    }
}
