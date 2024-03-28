/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank.rerank;

import org.apache.lucene.search.ScoreDoc;
import org.elasticsearch.action.search.SearchPhaseController;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.common.util.concurrent.ConcurrentCollections;
import org.elasticsearch.common.util.concurrent.CountDown;
import org.elasticsearch.search.query.QuerySearchResult;
import org.elasticsearch.search.rank.RankCoordinatorContext;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public abstract class RerankingRankCoordinatorContext extends RankCoordinatorContext {

    protected final String inferenceId = "some-model";
    protected final String inferenceText = "some-query-text";

    public RerankingRankCoordinatorContext(int size, int from, int windowSize, Client client) {
        super(size, from, windowSize, client);
    }

    @Override
    public SearchPhaseController.SortedTopDocs postQueryRank(
        List<QuerySearchResult> querySearchResults,
        SearchPhaseController.TopDocsStats topDocStats
    ) {
        List<FeatureRankDoc> rankDocs = new ArrayList<>();
        for (int i = 0; i < querySearchResults.size(); i++) {
            QuerySearchResult querySearchResult = querySearchResults.get(i);
            RankShardFeatureResult shardResult = (RankShardFeatureResult) querySearchResult.getRankShardResult();
            for (FeatureRankDoc frd : shardResult.featureRankDocs) {
                frd.shardIndex = i;
                rankDocs.add(frd);
            }
        }
        // no support for sort field atm
        // should pass needed info to make use of org.elasticsearch.action.search.SearchPhaseController.sortDocs?
        rankDocs.sort(Comparator.comparing((FeatureRankDoc doc) -> doc.score).reversed());
        FeatureRankDoc[] topResults = rankDocs.stream()
            .skip(from)
            .limit(Math.min(windowSize, rankDocs.size() - from))
            .toArray(FeatureRankDoc[]::new);

        assert topDocStats.fetchHits == 0;
        topDocStats.fetchHits = topResults.length;
        return new SearchPhaseController.SortedTopDocs(topResults, false, null, null, null, 0);
    }

    private Map<RankKey, String> extractFeatures(List<RankFeatureResult> rankSearchResults) {
        Map<RankKey, String> docFeatures = new LinkedHashMap<>(windowSize);
        for (RankFeatureResult rankFeatureResult : rankSearchResults) {
            RankShardFeatureResult shardResult = rankFeatureResult.shardResult();
            for (FeatureRankDoc doc : shardResult.featureRankDocs) {
                docFeatures.put(new RankKey(doc.doc, doc.shardIndex), doc.featureData);
            }
        }
        return docFeatures;
    }

    @Override
    public void reRank(List<RankFeatureResult> rankSearchResults, Consumer<ScoreDoc[]> onFinish) {
        Map<RankKey, String> features = extractFeatures(rankSearchResults);
        final List<Map<RankKey, String>> documentBatches = batches(features);
        final CountDown countDown = new CountDown(documentBatches.size());
        final Map<RankKey, FeatureRankDoc> rankMap = ConcurrentCollections.newConcurrentMap();
        for (Map<RankKey, String> batch : documentBatches) {
            final RankKey[] rankKeys = batch.keySet().toArray(new RankKey[0]);
            final Consumer<double[]> scoreConsumer = (scores) -> {
                assert scores.length == batch.size();
                for (int i = 0; i < scores.length; i++) {
                    RankKey rk = rankKeys[i];
                    float score = (float) scores[i];
                    rankMap.put(rk, new FeatureRankDoc(rk.doc(), score, rk.shardIndex()));
                }
            };
            computeUpdatedScores((List<String>) batch.values(), scoreConsumer, countDown, () -> {
                FeatureRankDoc[] sortedResults = rankMap.values().toArray(FeatureRankDoc[]::new);
                Arrays.sort(sortedResults, Comparator.comparing((FeatureRankDoc doc) -> doc.score).reversed());
                FeatureRankDoc[] topResults = new FeatureRankDoc[Math.min(size, sortedResults.length - from)];
                for (int rank = 0; rank < topResults.length; ++rank) {
                    topResults[rank] = sortedResults[rank];
                    topResults[rank].rank = rank + 1 + from;
                }
                onFinish.accept(topResults);
            });
        }
    }

    protected abstract void computeUpdatedScores(
        List<String> features,
        Consumer<double[]> scoreConsumer,
        CountDown countDown,
        Runnable onFinish
    );

    protected abstract List<Map<RankKey, String>> batches(Map<RankKey, String> docFeatures);

    @Override
    public boolean isRerank() {
        return true;
    }
}
