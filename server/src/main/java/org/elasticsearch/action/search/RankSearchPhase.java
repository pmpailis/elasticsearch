/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */
package org.elasticsearch.action.search;

import org.apache.lucene.search.ScoreDoc;
import org.elasticsearch.common.util.concurrent.AbstractRunnable;
import org.elasticsearch.search.SearchPhaseResult;
import org.elasticsearch.search.SearchShardTarget;
import org.elasticsearch.search.dfs.AggregatedDfs;
import org.elasticsearch.search.internal.ShardRankRequest;
import org.elasticsearch.search.rank.RankSearchResult;

import java.util.Collections;
import java.util.List;

/**
* ranksearchphase
 */
public final class RankSearchPhase extends SearchPhase {
    private final SearchPhaseContext context;
    private final SearchPhaseResults<SearchPhaseResult> resultConsumer;
    private final SearchPhaseResults<SearchPhaseResult> rankResults;

    private final AggregatedDfs aggregatedDfs;

    RankSearchPhase(SearchPhaseResults<SearchPhaseResult> resultConsumer, AggregatedDfs aggregatedDfs, SearchPhaseContext context) {
        super("rank-search");
        if (context.getNumShards() != resultConsumer.getNumShards()) {
            throw new IllegalStateException(
                "number of shards must match the length of the query results but doesn't:"
                    + context.getNumShards()
                    + "!="
                    + resultConsumer.getNumShards()
            );
        }
        this.context = context;
        this.resultConsumer = resultConsumer;
        this.aggregatedDfs = aggregatedDfs;
        this.rankResults = new ArraySearchPhaseResults<>(context.getNumShards());
        context.addReleasable(rankResults);
    }

    @Override
    public void run() {
        context.execute(new AbstractRunnable() {
            @Override
            protected void doRun() throws Exception {
                innerRun();
            }

            @Override
            public void onFailure(Exception e) {
                context.onPhaseFailure(RankSearchPhase.this, "", e);
            }
        });
    }

    private void executeNextPhase(SearchPhaseController.ReducedQueryPhase reducedQueryPhase) {
        SearchPhaseController.SortedTopDocs rerankedTopDocs = reducedQueryPhase.rankCoordinatorContext()
            .rank(
                rankResults.getAtomicArray().toArray(new RankSearchResult[context.getNumShards()]),
                reducedQueryPhase.sortedTopDocs(),
                reducedQueryPhase.topDocsStats()
            );
        reducedQueryPhase = new SearchPhaseController.ReducedQueryPhase(
            reducedQueryPhase.totalHits(),
            reducedQueryPhase.topDocsStats(),
            reducedQueryPhase.suggest(),
            reducedQueryPhase.aggregations(),
            reducedQueryPhase.profileBuilder(),
            rerankedTopDocs,
            reducedQueryPhase.sortValueFormats(),
            reducedQueryPhase.rankCoordinatorContext(),
            reducedQueryPhase.numReducePhases(),
            reducedQueryPhase.size(),
            reducedQueryPhase.from(),
            reducedQueryPhase.isEmptyResult()
        );
        context.executeNextPhase(this, new FetchSearchPhase(rankResults, aggregatedDfs, context, reducedQueryPhase));
    }

    private void innerRun() throws Exception {
        // we call firstPhaseResults within reduce
        SearchPhaseController.ReducedQueryPhase reducedQueryPhase = resultConsumer.reduce();
        boolean isRankSearch = reducedQueryPhase.rankCoordinatorContext() != null;
        if (isRankSearch) {
            ScoreDoc[] firstPhaseResults = reducedQueryPhase.sortedTopDocs().scoreDocs();
            final List<Integer>[] docIdsToLoad = SearchPhaseController.fillDocIdsToLoad(context.getNumShards(), firstPhaseResults);

            final CountedCollector<SearchPhaseResult> counter = new CountedCollector<>(
                rankResults,
                docIdsToLoad.length, // we count down every shard in the result no matter if we got any results or not
                () -> {
                    try {
                        executeNextPhase(reducedQueryPhase);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }, // we execute the next phase when all shards have been processed
                context
            );

            // next send a request to each shard
            for (int i = 0; i < docIdsToLoad.length; i++) {
                List<Integer> entry = docIdsToLoad[i];
                if (entry == null) {
                    counter.countDown();
                    continue;
                }
                SearchPhaseResult queryResult = resultConsumer.getAtomicArray().get(i);
                executeRankPhase(queryResult, counter, entry);
            }
        }
    }

    private void executeRankPhase(
        SearchPhaseResult queryResult,
        final CountedCollector<SearchPhaseResult> counter,
        final List<Integer> entry
    ) {
        final SearchShardTarget shardTarget = queryResult.getSearchShardTarget();
        final int shardIndex = queryResult.getShardIndex();
        context.getSearchTransport()
            .sendExecuteRank(
                context.getConnection(shardTarget.getClusterAlias(), shardTarget.getNodeId()),
                new ShardRankRequest(
                    context.getOriginalIndices(queryResult.getShardIndex()),
                    queryResult.getContextId(),
                    queryResult.getShardSearchRequest(),
                    entry,
                    Collections.emptyList()
                ),
                context.getTask(),
                new SearchActionListener<>(shardTarget, shardIndex) {
                    @Override
                    protected void innerOnResponse(RankSearchResult response) {
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
                        try {
                            // logger.debug(() -> "[" + contextId + "] Failed to execute fetch phase", e);
                            // progressListener.notifyFetchFailure(shardIndex, shardTarget, e);

                            counter.onFailure(shardIndex, shardTarget, e);
                        } finally {
                            // the search context might not be cleared on the node where the fetch was executed for example
                            // because the action was rejected by the thread pool. in this case we need to send a dedicated
                            // request to clear the search context.
                            // releaseIrrelevantSearchContext(queryResult.queryResult());
                        }
                    }
                }
            );
    }
}
