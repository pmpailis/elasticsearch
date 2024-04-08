/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank.rerank;

import org.elasticsearch.common.util.concurrent.CountDown;
import org.elasticsearch.search.rank.RankDoc.RankKey;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * A global reranker operating on the top `window_size` results from all shards, that provides a random order of the top hits.
 */
public class RandomOrderRankFeaturePhaseRankCoordinatorContext extends RerankingRankFeaturePhaseRankCoordinatorContext {

    public RandomOrderRankFeaturePhaseRankCoordinatorContext(int size, int from, int windowSize) {
        super(size, from, windowSize);
    }

    @Override
    protected void computeUpdatedScores(
        Map<RankKey, String> featureData,
        Consumer<double[]> scoreConsumer,
        CountDown countDown,
        Runnable onFinish
    ) {
        List<String> features = featureData.values().stream().toList();
        double[] scores = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            scores[i] = Math.random();
        }
        scoreConsumer.accept(scores);
        if (countDown.countDown()) {
            onFinish.run();
        }
    }

    // No need to do any batching here
    @Override
    protected List<Map<RankKey, String>> batches(Map<RankKey, String> docFeatures) {
        return Collections.singletonList(docFeatures);
    }

}
