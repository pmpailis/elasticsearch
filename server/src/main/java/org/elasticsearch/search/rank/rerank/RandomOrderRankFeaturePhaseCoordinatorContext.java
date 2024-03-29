/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank.rerank;

import org.elasticsearch.common.util.concurrent.CountDown;
import org.elasticsearch.search.rank.RankBuilder;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * A global reranker operating on the top `window_size` results from all shards, that provides a random order of the top hits.
 */
public class RandomOrderRankFeaturePhaseCoordinatorContext extends RerankingRankFeaturePhaseCoordinatorContext {

    public RandomOrderRankFeaturePhaseCoordinatorContext(int size, int from, int windowSize) {
        super(size, from, windowSize);
    }

    @Override
    protected void computeUpdatedScores(List<String> features, Consumer<double[]> scoreConsumer, CountDown countDown, Runnable onFinish) {
        double[] scores = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            scores[i] = Math.random();
        }
        scoreConsumer.accept(scores);
        if (countDown.countDown()) {
            onFinish.run();
        }
    }

    @Override
    protected List<Map<RankBuilder.RankKey, String>> batches(Map<RankBuilder.RankKey, String> docFeatures) {
        return Collections.singletonList(docFeatures);
    }

}
