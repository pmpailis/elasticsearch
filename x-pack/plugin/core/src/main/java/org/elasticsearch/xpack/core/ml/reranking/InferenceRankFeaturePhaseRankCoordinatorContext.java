/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.core.ml.reranking;

import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.ActionRequest;
import org.elasticsearch.action.ActionResponse;
import org.elasticsearch.action.ActionType;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.common.util.concurrent.CountDown;
import org.elasticsearch.search.rank.RankDoc;
import org.elasticsearch.search.rank.rerank.RerankingRankFeaturePhaseRankCoordinatorContext;

import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public abstract class InferenceRankFeaturePhaseRankCoordinatorContext<Request extends ActionRequest, Response extends ActionResponse>
    extends RerankingRankFeaturePhaseRankCoordinatorContext {

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

    protected abstract Request request(List<String> docFeatures);

    protected abstract ActionType<Response> action();

    protected abstract double[] extractScoresFromResponse(Response response);

    @Override
    protected List<Map<RankDoc.RankKey, String>> batches(Map<RankDoc.RankKey, String> docFeatures) {
        return List.of(docFeatures);
    }

    @Override
    protected void computeUpdatedScores(
        Map<RankDoc.RankKey, String> docFeatures,
        Consumer<double[]> scoreConsumer,
        CountDown countDown,
        Runnable onFinish
    ) {
        List<String> features = docFeatures.values().stream().toList();
        final ActionListener<Response> actionListener = listener(scoreConsumer, countDown, onFinish);
        Request req = request(features);
        ActionType<Response> action = action();
        client.execute(action, req, actionListener);
    }

    private ActionListener<Response> listener(Consumer<double[]> scoreConsumer, CountDown countDown, Runnable onFinish) {
        return new ActionListener<>() {
            @Override
            public void onResponse(Response response) {
                try {
                    if (response != null) {
                        double[] scores = extractScoresFromResponse(response);
                        scoreConsumer.accept(scores);
                    }
                } finally {
                    if (response != null) {
                        response.decRef();
                    }
                    if (countDown.countDown()) {
                        onFinish.run();
                    }
                }
            }

            @Override
            public void onFailure(Exception e) {
                try {
                    System.out.println(e.getMessage());
                } finally {
                    if (countDown.countDown()) {
                        onFinish.run();
                    }
                }
            }
        };
    }
}
