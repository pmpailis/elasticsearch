/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.core.ml.rerank;

import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.ActionType;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.inference.InferenceServiceResults;
import org.elasticsearch.inference.InputType;
import org.elasticsearch.inference.TaskType;
import org.elasticsearch.search.rank.context.RankFeaturePhaseRankCoordinatorContext;
import org.elasticsearch.search.rank.feature.RankFeatureDoc;
import org.elasticsearch.xpack.core.inference.action.InferenceAction;
import org.elasticsearch.xpack.core.inference.results.RankedDocsResults;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class TextSimilarityRankFeaturePhaseRankCoordinatorContext extends RankFeaturePhaseRankCoordinatorContext {

    private final String inferenceText;
    private final String inferenceId;
    private final Client client;
    private final String field;

    public TextSimilarityRankFeaturePhaseRankCoordinatorContext(
        int size,
        int from,
        int windowSize,
        Client client,
        String inferenceId,
        String inferenceText,
        float minScore,
        String field,
        RankFeaturePhaseRankCoordinatorContext delegate
    ) {
        super(size, from, windowSize, delegate);
        this.client = client;
        this.inferenceId = inferenceId;
        this.inferenceText = inferenceText;
        this.field = field;
    }

    protected InferenceAction.Request generateRequest(List<String> docFeatures) {
        return new InferenceAction.Request(
            TaskType.RERANK,
            inferenceId,
            inferenceText,
            docFeatures,
            Map.of(),
            InputType.SEARCH,
            InferenceAction.Request.DEFAULT_TIMEOUT
        );
    }

    protected ActionType<InferenceAction.Response> actionType() {
        return InferenceAction.INSTANCE;
    }

    protected float[] extractScoresFromResponse(InferenceAction.Response response) {
        InferenceServiceResults results = response.getResults();
        assert results instanceof RankedDocsResults;

        List<RankedDocsResults.RankedDoc> rankedDocs = ((RankedDocsResults) results).getRankedDocs();
        float[] scores = new float[rankedDocs.size()];
        for (RankedDocsResults.RankedDoc rankedDoc : rankedDocs) {
            scores[rankedDoc.index()] = rankedDoc.relevanceScore();
        }

        return scores;
    }

    @Override
    protected void doComputeScores(RankFeatureDoc[] featureDocs, ActionListener<RankFeatureDoc[]> rankDocs) {
        // Wrap the provided rankListener to an ActionListener that would handle the response from the inference service
        // and then pass the results
        final ActionListener<InferenceAction.Response> actionListener = rankDocs.delegateFailureAndWrap((l, r) -> {
            float[] scores = extractScoresFromResponse(r);
            assert scores.length == featureDocs.length;
            for (int i = 0; i < featureDocs.length; i++) {
                featureDocs[i].score = scores[i];
            }
            l.onResponse(featureDocs);
        });

        List<String> featureData = Arrays.stream(featureDocs).map(x -> x.featureData.get(field).toString()).toList();
        InferenceAction.Request request = generateRequest(featureData);
        try {
            ActionType<InferenceAction.Response> action = actionType();
            client.execute(action, request, actionListener);
        } finally {
            if (request != null) {
                request.decRef();
            }
        }
    }
}