/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank.rerank;

import org.apache.lucene.search.ScoreDoc;
import org.elasticsearch.search.rank.context.RankFeaturePhaseRankCoordinatorContext;
import org.elasticsearch.search.rank.feature.RankFeatureDoc;
import org.elasticsearch.search.rank.feature.RankFeatureResult;
import org.elasticsearch.search.rank.feature.RankFeatureShardResult;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * Abstract class that provides the logic for reranking the top `rank_window_size` results from all shards,
 * based on the provided {@link RankFeatureResult} from the {@link org.elasticsearch.action.search.RankFeaturePhase} execution.
 * The computation is done in an async manner so that we can also facilitate potential call to external services if needed.
 */
public abstract class RerankingRankFeaturePhaseRankCoordinatorContext extends RankFeaturePhaseRankCoordinatorContext {

    public RerankingRankFeaturePhaseRankCoordinatorContext(int size, int from, int windowSize) {
        super(size, from, windowSize);
    }

    /**
     * This method is responsible for computing the updated scores for a list of feature data. Once done, the `onFinish` should be called
     * to continue execution to the next phase.
     */
    protected abstract void computeScores(RankFeatureDoc[] featureDocs, BiConsumer<Integer, Float> scoreConsumer, Runnable onFinish);

    /**
     * This method is responsible for ranking the global results based on the provided rank feature results from each shard.
     * <p>
     * We first start by extracting ordered feature data accessed through a {@code List<RankFeatureDoc>}
     * from the provided rankSearchResults, and then compute the updated scores for each batch of documents.
     * Once all the scores have been computed, we sort the results, perform any pagination needed, and then call the `onFinish` consumer
     * with the final array of {@link ScoreDoc} results.
     *
     * @param rankSearchResults a list of rank feature results from each shard
     * @param onFinish          a consumer to be called once the global ranking is complete
     */
    @Override
    public void rankGlobalResults(List<RankFeatureResult> rankSearchResults, Consumer<ScoreDoc[]> onFinish) {
        // extract feature data from each shard rank-feature phase result
        RankFeatureDoc[] featureDocs = extractFeatureDocs(rankSearchResults);

        // once we have an update score, provide a function to store that score based on the index of the feature doc
        final BiConsumer<Integer, Float> scoreConsumer = (index, score) -> {
            assert index >= 0 && index < featureDocs.length;
            featureDocs[index].score = score;
        };

        // compute the scores for the feature docs based on the extracted data, and once done
        // generate the final `topResults` paginated results, and pass them to fetch phase through the
        // `onFinish` consumer.
        computeScores(featureDocs, scoreConsumer, () -> {
            Arrays.sort(featureDocs, Comparator.comparing((RankFeatureDoc doc) -> doc.score).reversed());
            RankFeatureDoc[] topResults = new RankFeatureDoc[Math.max(0, Math.min(size, featureDocs.length - from))];
            for (int rank = 0; rank < topResults.length; ++rank) {
                topResults[rank] = featureDocs[from + rank];
                topResults[rank].rank = from + rank + 1;
            }
            // and call the parent onFinish consumer with the final `ScoreDoc[]` results.
            onFinish.accept(topResults);
        });
    }

    private RankFeatureDoc[] extractFeatureDocs(List<RankFeatureResult> rankSearchResults) {
        List<RankFeatureDoc> docFeatures = new ArrayList<>();
        for (RankFeatureResult rankFeatureResult : rankSearchResults) {
            RankFeatureShardResult shardResult = rankFeatureResult.shardResult();
            docFeatures.addAll(Arrays.stream(shardResult.rankFeatureDocs).toList());
        }
        return docFeatures.toArray(new RankFeatureDoc[0]);
    }
}
