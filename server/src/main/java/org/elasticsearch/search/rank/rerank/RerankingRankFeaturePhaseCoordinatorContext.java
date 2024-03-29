/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank.rerank;

import org.apache.lucene.search.ScoreDoc;
import org.elasticsearch.common.util.concurrent.ConcurrentCollections;
import org.elasticsearch.common.util.concurrent.CountDown;
import org.elasticsearch.search.rank.RankBuilder;
import org.elasticsearch.search.rank.RankFeaturePhaseCoordinatorContext;
import org.elasticsearch.search.rank.feature.RankFeatureDoc;
import org.elasticsearch.search.rank.feature.RankFeatureResult;
import org.elasticsearch.search.rank.feature.RankFeatureShardResult;

import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public abstract class RerankingRankFeaturePhaseCoordinatorContext extends RankFeaturePhaseCoordinatorContext {

    public RerankingRankFeaturePhaseCoordinatorContext(int size, int from, int windowSize) {
        super(size, from, windowSize);
    }

    protected abstract void computeUpdatedScores(
        List<String> features,
        Consumer<double[]> scoreConsumer,
        CountDown countDown,
        Runnable onFinish
    );

    protected abstract List<Map<RankBuilder.RankKey, String>> batches(Map<RankBuilder.RankKey, String> docFeatures);

    @Override
    public void rankGlobalResults(List<RankFeatureResult> rankSearchResults, Consumer<ScoreDoc[]> onFinish) {
        Map<RankBuilder.RankKey, String> features = extractFeatures(rankSearchResults);
        final List<Map<RankBuilder.RankKey, String>> documentBatches = batches(features);
        final CountDown countDown = new CountDown(documentBatches.size());
        final Map<RankBuilder.RankKey, RankFeatureDoc> rankMap = ConcurrentCollections.newConcurrentMap();
        for (Map<RankBuilder.RankKey, String> batch : documentBatches) {
            final RankBuilder.RankKey[] rankKeys = batch.keySet().toArray(new RankBuilder.RankKey[0]);
            final Consumer<double[]> scoreConsumer = (scores) -> {
                assert scores.length == batch.size();
                for (int i = 0; i < scores.length; i++) {
                    RankBuilder.RankKey rk = rankKeys[i];
                    float score = (float) scores[i];
                    rankMap.put(rk, new RankFeatureDoc(rk.doc(), score, rk.shardIndex()));
                }
            };
            computeUpdatedScores(batch.values().stream().toList(), scoreConsumer, countDown, () -> {
                RankFeatureDoc[] sortedResults = rankMap.values().toArray(RankFeatureDoc[]::new);
                Arrays.sort(sortedResults, Comparator.comparing((RankFeatureDoc doc) -> doc.score).reversed());
                RankFeatureDoc[] topResults = new RankFeatureDoc[Math.max(0, Math.min(size, sortedResults.length - from))];
                for (int rank = 0; rank < topResults.length; ++rank) {
                    topResults[rank] = sortedResults[from + rank];
                    topResults[rank].rank = from + rank + 1;
                }
                onFinish.accept(topResults);
            });
        }
    }

    private Map<RankBuilder.RankKey, String> extractFeatures(List<RankFeatureResult> rankSearchResults) {
        Map<RankBuilder.RankKey, String> docFeatures = new LinkedHashMap<>(windowSize);
        for (RankFeatureResult rankFeatureResult : rankSearchResults) {
            RankFeatureShardResult shardResult = rankFeatureResult.shardResult();
            for (RankFeatureDoc doc : shardResult.rankFeatureDocs) {
                docFeatures.put(new RankBuilder.RankKey(doc.doc, doc.shardIndex), doc.featureData);
            }
        }
        return docFeatures;
    }
}
