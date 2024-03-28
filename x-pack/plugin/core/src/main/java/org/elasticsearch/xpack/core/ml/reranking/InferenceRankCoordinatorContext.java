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
import org.elasticsearch.search.rank.rerank.RerankingRankCoordinatorContext;

import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public abstract class InferenceRankCoordinatorContext<Request extends ActionRequest, Response extends ActionResponse> extends
    RerankingRankCoordinatorContext {

    protected final String inferenceId = "some-model";
    protected final String inferenceText = "some-query-text";

    public InferenceRankCoordinatorContext(int size, int from, int windowSize, Client client) {
        super(size, from, windowSize, client);
    }

    protected abstract Request request(List<String> docFeatures);

    protected abstract ActionType<Response> action();

    protected abstract double[] extractScoresFromResponse(Response response);

    @Override
    protected List<Map<RankKey, String>> batches(Map<RankKey, String> docFeatures) {
        return List.of(docFeatures);
    }

    @Override
    protected void computeUpdatedScores(List<String> features, Consumer<double[]> scoreConsumer, CountDown countDown, Runnable onFinish) {
        final ActionListener<Response> actionListener = listener(scoreConsumer, countDown, onFinish);
        Request req = request(features);
        ActionType<Response> action = action();
        client.execute(action, req, actionListener);
    }

    private ActionListener<Response> listener(Consumer<double[]> scoreConsumer, CountDown countDown, Runnable onFinish) {
        return new ActionListener<>() {
            @Override
            public void onResponse(Response response) {
                assert response != null;
                double[] scores = extractScoresFromResponse(response);
                scoreConsumer.accept(scores);
                if (countDown.countDown()) {
                    onFinish.run();
                }
            }

            @Override
            public void onFailure(Exception e) {
                System.out.println(e.getMessage());
                if (countDown.countDown()) {
                    onFinish.run();
                }
            }
        };
    }
}
