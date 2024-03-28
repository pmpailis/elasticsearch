/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */
package org.elasticsearch.action.search;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.common.util.concurrent.AbstractRunnable;
import org.elasticsearch.search.SearchPhaseResult;
import org.elasticsearch.search.SearchShardTarget;
import org.elasticsearch.search.dfs.AggregatedDfs;
import org.elasticsearch.search.internal.ShardSearchContextId;
import org.elasticsearch.search.rank.rerank.RankFeatureResult;
import org.elasticsearch.search.rank.rerank.RankShardFeatureRequest;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * This class is responsible for executing the rank phase of the search.
 * It starts by retrieving num_shards * window_size results from the query phase and
 * reducing them to a global list of ranked `window_size` results.
 * It then sends a request to each of the shards in this list to fetch all needed feature info (e.g. field data)
 * in order to do the global reranking.
 */
public final class RankSearchPhase extends SearchPhase {

    private static final Logger logger = LogManager.getLogger(RankSearchPhase.class);
    private final SearchPhaseContext context;
    private final SearchPhaseResults<SearchPhaseResult> queryPhaseResults;
    private final SearchPhaseResults<SearchPhaseResult> rankPhaseResults;
    private final SearchProgressListener progressListener;

    private final AggregatedDfs aggregatedDfs;

    RankSearchPhase(SearchPhaseResults<SearchPhaseResult> queryPhaseResults, AggregatedDfs aggregatedDfs, SearchPhaseContext context) {
        super("rank-search");
        if (context.getNumShards() != queryPhaseResults.getNumShards()) {
            throw new IllegalStateException(
                "number of shards must match the length of the query results but doesn't:"
                    + context.getNumShards()
                    + "!="
                    + queryPhaseResults.getNumShards()
            );
        }
        this.context = context;
        this.queryPhaseResults = queryPhaseResults;
        this.aggregatedDfs = aggregatedDfs;
        this.rankPhaseResults = new ArraySearchPhaseResults<>(context.getNumShards());
        context.addReleasable(rankPhaseResults);
        this.progressListener = context.getTask().getProgressListener();
    }

    @Override
    public void run() {
        context.execute(new AbstractRunnable() {
            @Override
            protected void doRun() throws Exception {
                // we need to reduce the results at this point instead of fetch phase, so we fork this process similarly to how
                // was set up at FetchSearchPhase.

                // we do the heavy lifting in this inner run method where we reduce aggs etc. that's why we fork this phase
                // off immediately instead of forking when we send back the response to the user since there we only need
                // to merge together the fetched results which is a linear operation.
                innerRun();
            }

            @Override
            public void onFailure(Exception e) {
                context.onPhaseFailure(RankSearchPhase.this, "", e);
            }
        });
    }

    private void onPhaseDone(SearchPhaseController.ReducedQueryPhase reducedQueryPhase) {
        reducedQueryPhase.rankCoordinatorContext()
            .reRank(rankPhaseResults.getAtomicArray().asList().stream().map(SearchPhaseResult::rankFeatureResult).toList(), (scoreDocs) -> {
                SearchPhaseController.ReducedQueryPhase rerankedReducedQueryPhase = new SearchPhaseController.ReducedQueryPhase(
                    reducedQueryPhase.totalHits(),
                    reducedQueryPhase.fetchHits(),
                    Arrays.stream(scoreDocs).map(x -> x.score).max(Comparator.comparingDouble(x -> x)).orElse(Float.NaN),
                    reducedQueryPhase.timedOut(),
                    reducedQueryPhase.terminatedEarly(),
                    reducedQueryPhase.suggest(),
                    reducedQueryPhase.aggregations(),
                    reducedQueryPhase.profileBuilder(),
                    new SearchPhaseController.SortedTopDocs(scoreDocs, false, null, null, null, 0),
                    reducedQueryPhase.sortValueFormats(),
                    reducedQueryPhase.rankCoordinatorContext(),
                    reducedQueryPhase.numReducePhases(),
                    reducedQueryPhase.size(),
                    reducedQueryPhase.from(),
                    reducedQueryPhase.isEmptyResult()
                );
                moveToNextPhase(rankPhaseResults, rerankedReducedQueryPhase);
            });
    }

    private void moveToNextPhase(
        SearchPhaseResults<SearchPhaseResult> phaseResults,
        SearchPhaseController.ReducedQueryPhase reducedQueryPhase
    ) {
        context.executeNextPhase(this, new FetchSearchPhase(phaseResults, aggregatedDfs, context, reducedQueryPhase));
    }

    private void innerRun() throws Exception {
        SearchPhaseController.ReducedQueryPhase reducedQueryPhase = queryPhaseResults.reduce();
        boolean isRerankSearch = reducedQueryPhase.rankCoordinatorContext() != null && reducedQueryPhase.rankCoordinatorContext().isRerank();
        if (isRerankSearch) {
            SearchPhaseController.SortedTopDocs firstPhaseResults = reducedQueryPhase.sortedTopDocs();
            final List<Integer>[] docIdsToLoad = SearchPhaseController.fillDocIdsToLoad(
                context.getNumShards(),
                firstPhaseResults.scoreDocs()
            );
            final CountedCollector<SearchPhaseResult> counter = new CountedCollector<>(
                rankPhaseResults,
                context.getNumShards(),
                () -> onPhaseDone(reducedQueryPhase),
                context
            );

            // we send out a request to each shard in order to fetch the needed feature info
            for (int i = 0; i < docIdsToLoad.length; i++) {
                List<Integer> entry = docIdsToLoad[i];
                if (entry == null || entry.isEmpty()) {
                    counter.countDown();
                    continue;
                }
                SearchPhaseResult queryResult = queryPhaseResults.getAtomicArray().get(i);
                executeFeatureShardPhase(queryResult, counter, entry);
            }
        } else {
            moveToNextPhase(queryPhaseResults, reducedQueryPhase);
        }
    }

    private void executeFeatureShardPhase(
        SearchPhaseResult queryResult,
        final CountedCollector<SearchPhaseResult> counter,
        final List<Integer> entry
    ) {
        final SearchShardTarget shardTarget = queryResult.queryResult().getSearchShardTarget();
        final ShardSearchContextId contextId = queryResult.queryResult().getContextId();
        final int shardIndex = queryResult.getShardIndex();
        context.getSearchTransport()
            .sendExecuteRankFeature(
                context.getConnection(shardTarget.getClusterAlias(), shardTarget.getNodeId()),
                new RankShardFeatureRequest(
                    context.getOriginalIndices(queryResult.getShardIndex()),
                    queryResult.getContextId(),
                    queryResult.getShardSearchRequest(),
                    entry,
                    "body" // TODO: retrieve from request
                ),
                context.getTask(),
                new SearchActionListener<>(shardTarget, shardIndex) {
                    @Override
                    protected void innerOnResponse(RankFeatureResult response) {
                        try {
                            if (response != null) {
                                counter.onResult(response);
                            } else {
                                counter.countDown();
                            }
                        } catch (Exception e) {
                            context.onPhaseFailure(RankSearchPhase.this, "", e);
                        }
                    }

                    @Override
                    public void onFailure(Exception e) {
                        logger.debug(() -> "[" + contextId + "] Failed to execute rank phase", e);
                        progressListener.notifyFetchFailure(shardIndex, shardTarget, e);
                        counter.onFailure(shardIndex, shardTarget, e);
                    }
                }
            );
    }
}
