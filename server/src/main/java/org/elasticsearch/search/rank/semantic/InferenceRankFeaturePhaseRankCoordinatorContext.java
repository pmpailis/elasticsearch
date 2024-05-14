/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank.semantic;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.ActionRequest;
import org.elasticsearch.action.ActionResponse;
import org.elasticsearch.action.ActionType;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.search.rank.feature.RankFeatureDoc;
import org.elasticsearch.search.rank.rerank.RerankingRankFeaturePhaseRankCoordinatorContext;

import java.util.Arrays;
import java.util.List;
import java.util.function.BiConsumer;

public abstract class InferenceRankFeaturePhaseRankCoordinatorContext<Request extends ActionRequest, Response extends ActionResponse>
    extends RerankingRankFeaturePhaseRankCoordinatorContext {

    private static final Logger logger = LogManager.getLogger(InferenceRankFeaturePhaseRankCoordinatorContext.class);
    protected final String inferenceId;
    protected final String inferenceText;

    protected final Client client;

    public InferenceRankFeaturePhaseRankCoordinatorContext(
        int size,
        int from,
        int windowSize,
        Client client,
        String inferenceId,
        String inferenceText
    ) {
        super(size, from, windowSize);
        this.client = client;
        this.inferenceId = inferenceId;
        this.inferenceText = inferenceText;
    }

    protected abstract Request generateRequest(List<String> docFeatures);

    protected abstract ActionType<Response> actionType();

    protected abstract float[] extractScoresFromResponse(Response response);

    @Override
    protected void computeScores(RankFeatureDoc[] featureDocs, BiConsumer<Integer, Float> scoreConsumer, Runnable onFinish) {
        final ActionListener<Response> actionListener = ActionListener.runAfter(new ActionListener<>() {
            @Override
            public void onResponse(Response response) {
                try {
                    if (response != null) {
                        float[] scores = extractScoresFromResponse(response);
                        for (int i = 0; i < scores.length; i++) {
                            scoreConsumer.accept(i, scores[i]);
                        }
                    }
                } finally {
                    if (response != null) {
                        response.decRef();
                    }
                }
            }

            @Override
            public void onFailure(Exception e) {
                assert false : e;
                logger.warn(
                    () -> "failed to generate response for inferenceId: [" + inferenceId + "] and inferenceText: [" + inferenceText + "].",
                    e
                );
            }
        }, onFinish);
        List<String> featureData = Arrays.stream(featureDocs).map(x -> x.featureData).toList();
        Request request = generateRequest(featureData);
        ActionType<Response> action = actionType();
        client.execute(action, request, actionListener);
    }
}
