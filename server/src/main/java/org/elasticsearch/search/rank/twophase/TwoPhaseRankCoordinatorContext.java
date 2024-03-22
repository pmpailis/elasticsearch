/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.search.rank.twophase;

import org.apache.lucene.search.ScoreDoc;
import org.elasticsearch.action.search.SearchPhaseController.SortedTopDocs;
import org.elasticsearch.action.search.SearchPhaseController.TopDocsStats;
import org.elasticsearch.search.SearchPhaseResult;
import org.elasticsearch.search.query.QuerySearchResult;
import org.elasticsearch.search.rank.RankCoordinatorContext;
import org.elasticsearch.search.rank.RankDoc;
import org.elasticsearch.search.rank.RankShardResult;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Ranks and decorates search hits for RRF results on the coordinator.
 */
public class TwoPhaseRankCoordinatorContext extends RankCoordinatorContext {

    public TwoPhaseRankCoordinatorContext(int size, int from, int windowSize) {
        super(size, from, windowSize);
    }

    // used for faster hash lookup in a map of ranked documents
    protected record RankKey(int doc, int shardIndex) {}

    @Override
    public SortedTopDocs rank(SearchPhaseResult[] shardRankResults, SortedTopDocs topDocs, TopDocsStats topDocsStats) {
        List<TwoPhaseRankDoc> rankDocs = new ArrayList<>();
        for (SearchPhaseResult result : shardRankResults) {
            RankShardResult rsr = result.rankSearchResult().shardResult();
            assert rsr instanceof TwoPhaseRankShardResult;
            Collections.addAll(rankDocs, ((TwoPhaseRankShardResult) rsr).reRankedDocs);
        }
        rankDocs.sort(Comparator.comparing(TwoPhaseRankDoc::getSecondPhaseScore).reversed());

        TwoPhaseRankDoc[] topResults = new TwoPhaseRankDoc[Math.min(size, rankDocs.size() - from)];
        for (int rank = 0; rank < topResults.length; ++rank) {
            TwoPhaseRankDoc scoreDoc = rankDocs.get(from + rank);
            topResults[rank] = new TwoPhaseRankDoc(scoreDoc.doc, scoreDoc.score, scoreDoc.shardIndex, scoreDoc.secondPhaseScore);
            topResults[rank].rank = from + rank;
            topResults[rank].score = scoreDoc.secondPhaseScore;
        }
        // update fetch hits for the fetch phase, so we gather any additional
        // information required just like a standard query
        assert topDocsStats.fetchHits == 0;
        topDocsStats.fetchHits = topResults.length;
        topDocsStats.setMaxScore(topResults[0].score);

        // return the top results where sort, collapse fields,
        // and completion suggesters are not allowed
        return new SortedTopDocs(topResults, false, null, null, null, 0);
    }

    @Override
    public SortedTopDocs firstPhaseRank(List<QuerySearchResult> querySearchResults, TopDocsStats topDocsStats) {
        // for first phase ranking, all the coordinator does is to sort the docs based on the initial score,
        // and keep the top "window size" docs for the second phase
        List<RankDoc> rankDocs = new ArrayList<>();
        for (QuerySearchResult qsr : querySearchResults) {
            RankShardResult rsr = qsr.getRankShardResult();
            assert rsr instanceof TwoPhaseRankShardResult;
            for (RankDoc rd : ((TwoPhaseRankShardResult) rsr).reRankedDocs) {
                rd.shardIndex = qsr.getShardIndex();
                rankDocs.add(rd);
            }
        }
        List<RankDoc> rankedDocs = rankDocs.stream().sorted(Comparator.comparing(RankDoc::getScore).reversed()).limit(windowSize).toList();
        return new SortedTopDocs(rankedDocs.toArray(new ScoreDoc[0]), false, null, null, null, 0);
    }
}
