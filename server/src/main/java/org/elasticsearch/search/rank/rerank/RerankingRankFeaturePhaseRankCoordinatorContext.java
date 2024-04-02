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
import org.elasticsearch.search.rank.RankDoc.RankKey;
import org.elasticsearch.search.rank.RankFeaturePhaseRankCoordinatorContext;
import org.elasticsearch.search.rank.feature.RankFeatureDoc;
import org.elasticsearch.search.rank.feature.RankFeatureResult;
import org.elasticsearch.search.rank.feature.RankFeatureShardResult;

import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Abstract class that provides the logic for reranking the top `window_size` results from all shards,
 * based on the provided {@link RankFeatureResult} from the {@link org.elasticsearch.action.search.RankFeaturePhase} execution.
 *
 * The computation is done in an async manner so that we can also facilitate potential call to external services if needed.
 */
public abstract class RerankingRankFeaturePhaseRankCoordinatorContext extends RankFeaturePhaseRankCoordinatorContext {

    public RerankingRankFeaturePhaseRankCoordinatorContext(int size, int from, int windowSize) {
        super(size, from, windowSize);
    }

    /**
     * This method is responsible for computing the updated scores for a list of feature data. It also provides a {@link CountDown counter}
     * that is expected to execute the `onFinish` runnable once all the scores have been computed.
     */
    protected abstract void computeUpdatedScores(
        List<String> features,
        Consumer<double[]> scoreConsumer,
        CountDown countDown,
        Runnable onFinish
    );

    /**
     * Perform any batching of the input feature data if needed
     */
    protected abstract List<Map<RankKey, String>> batches(Map<RankKey, String> docFeatures);

    /**
     * This method is responsible for ranking the global results based on the provided rank feature results from each shard.
     *
     * We first start by extracting ordered feature data accessed through a {@link LinkedHashMap<RankKey, String>}
     * from the provided rankSearchResults, and then compute the updated scores for each batch of documents.
     * Once all the scores have been computed, we sort the results, perform any pagination needed, and then call the `onFinish` consumer with the final array of {@link ScoreDoc} results.
     *
     * @param rankSearchResults a list of rank feature results from each shard
     * @param onFinish a consumer to be called once the global ranking is complete
     */
    @Override
    public void rankGlobalResults(List<RankFeatureResult> rankSearchResults, Consumer<ScoreDoc[]> onFinish) {
        Map<RankKey, String> features = extractFeatures(rankSearchResults);
        final List<Map<RankKey, String>> documentBatches = batches(features);
        // We initialize a CountDown so that we know when all the scores have been computed
        final CountDown countDown = new CountDown(documentBatches.size());
        final Map<RankKey, RankFeatureDoc> rankMap = ConcurrentCollections.newConcurrentMap();
        for (Map<RankKey, String> batch : documentBatches) {
            final RankKey[] rankKeys = batch.keySet().toArray(new RankKey[0]);
            // Here we initialize a consumer with the expectation that the order of the scores will match the order of the rankKeys provided
            final Consumer<double[]> scoreConsumer = (scores) -> {
                assert scores.length == batch.size();
                for (int i = 0; i < scores.length; i++) {
                    RankKey rk = rankKeys[i];
                    float score = (float) scores[i];
                    rankMap.put(rk, new RankFeatureDoc(rk.doc(), score, rk.shardIndex()));
                }
            };
            computeUpdatedScores(batch.values().stream().toList(), scoreConsumer, countDown, () -> {
                // Once we're doing computing all scores, we want to globally sort the results
                RankFeatureDoc[] sortedResults = rankMap.values().toArray(RankFeatureDoc[]::new);
                Arrays.sort(sortedResults, Comparator.comparing((RankFeatureDoc doc) -> doc.score).reversed());
                // perform any needed pagination
                RankFeatureDoc[] topResults = new RankFeatureDoc[Math.max(0, Math.min(size, sortedResults.length - from))];
                for (int rank = 0; rank < topResults.length; ++rank) {
                    topResults[rank] = sortedResults[from + rank];
                    topResults[rank].rank = from + rank + 1;
                }
                // and cal the parent onFinish consumer with the final `ScoreDoc[]` results.
                onFinish.accept(topResults);
            });
        }
    }

    private Map<RankKey, String> extractFeatures(List<RankFeatureResult> rankSearchResults) {
        Map<RankKey, String> docFeatures = new LinkedHashMap<>(windowSize);
        for (RankFeatureResult rankFeatureResult : rankSearchResults) {
            RankFeatureShardResult shardResult = rankFeatureResult.shardResult();
            for (RankFeatureDoc doc : shardResult.rankFeatureDocs) {
                docFeatures.put(new RankKey(doc.doc, doc.shardIndex), doc.featureData);
            }
        }
        return docFeatures;
    }
}
