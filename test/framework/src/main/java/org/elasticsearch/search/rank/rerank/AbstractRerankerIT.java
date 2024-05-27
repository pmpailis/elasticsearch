/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank.rerank;

import org.elasticsearch.action.search.SearchPhaseExecutionException;
import org.elasticsearch.cluster.metadata.IndexMetadata;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.rank.RankBuilder;
import org.elasticsearch.test.ESIntegTestCase;

import java.util.Arrays;
import java.util.Collection;

import static org.elasticsearch.index.query.QueryBuilders.boolQuery;
import static org.elasticsearch.index.query.QueryBuilders.constantScoreQuery;
import static org.elasticsearch.index.query.QueryBuilders.matchQuery;
import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertHitCount;
import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertNoFailuresAndResponse;
import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertResponse;
import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.hasId;
import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.hasRank;

@ESIntegTestCase.ClusterScope(minNumDataNodes = 3)
public abstract class AbstractRerankerIT extends ESIntegTestCase {

    public void testRerankerNoExceptions() throws Exception {
        final String indexName = "test_index";
        final String rankFeatureField = "rankFeatureField";
        final String searchField = "searchField";
        final int rankWindowSize = 10;

        createIndex(indexName);
        indexRandom(
            true,
            prepareIndex(indexName).setId("1").setSource(rankFeatureField, 0.1, searchField, "A"),
            prepareIndex(indexName).setId("2").setSource(rankFeatureField, 0.2, searchField, "B"),
            prepareIndex(indexName).setId("3").setSource(rankFeatureField, 0.3, searchField, "C"),
            prepareIndex(indexName).setId("4").setSource(rankFeatureField, 0.4, searchField, "D"),
            prepareIndex(indexName).setId("5").setSource(rankFeatureField, 0.5, searchField, "E")
        );

        assertNoFailuresAndResponse(
            prepareSearch().setQuery(
                boolQuery().should(constantScoreQuery(matchQuery(searchField, "A")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "B")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "C")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "D")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "E")).boost(randomFloat()))
            )
                .setRankBuilder(getRankBuilder(rankWindowSize, rankFeatureField))
                .addFetchField(searchField)
                .setTrackTotalHits(true)
                .setAllowPartialSearchResults(true)
                .setSize(10),
            response -> {
                assertHitCount(response, 5L);
                int rank = 1;
                for (SearchHit searchHit : response.getHits().getHits()) {
                    assertThat(searchHit, hasId(String.valueOf(5 - (rank - 1))));
                    assertEquals(searchHit.getScore(), (0.5f - ((rank - 1) * 0.1f)), 1e-5f);
                    assertThat(searchHit, hasRank(rank));
                    assertNotNull(searchHit.getFields().get(searchField));
                    rank++;
                }
            }
        );
    }

    public void testRerankerPagination() throws Exception {
        final String indexName = "test_index";
        final String rankFeatureField = "rankFeatureField";
        final String searchField = "searchField";
        final int rankWindowSize = 10;

        createIndex(indexName);
        indexRandom(
            true,
            prepareIndex(indexName).setId("1").setSource(rankFeatureField, 0.1, searchField, "A"),
            prepareIndex(indexName).setId("2").setSource(rankFeatureField, 0.2, searchField, "B"),
            prepareIndex(indexName).setId("3").setSource(rankFeatureField, 0.3, searchField, "C"),
            prepareIndex(indexName).setId("4").setSource(rankFeatureField, 0.4, searchField, "D"),
            prepareIndex(indexName).setId("5").setSource(rankFeatureField, 0.5, searchField, "E")
        );

        assertResponse(
            prepareSearch().setQuery(
                boolQuery().should(constantScoreQuery(matchQuery(searchField, "A")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "B")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "C")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "D")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "E")).boost(randomFloat()))
            )
                .setRankBuilder(getRankBuilder(rankWindowSize, rankFeatureField))
                .addFetchField(searchField)
                .setTrackTotalHits(true)
                .setAllowPartialSearchResults(true)
                .setSize(2)
                .setFrom(2),
            response -> {
                assertHitCount(response, 5L);
                int rank = 3;
                for (SearchHit searchHit : response.getHits().getHits()) {
                    assertThat(searchHit, hasId(String.valueOf(5 - (rank - 1))));
                    assertEquals(searchHit.getScore(), (0.5f - ((rank - 1) * 0.1f)), 1e-5f);
                    assertThat(searchHit, hasRank(rank));
                    assertNotNull(searchHit.getFields().get(searchField));
                    rank++;
                }
            }
        );
    }

    public void testRerankerPaginationOutsideOfBounds() throws Exception {
        final String indexName = "test_index";
        final String rankFeatureField = "rankFeatureField";
        final String searchField = "searchField";
        final int rankWindowSize = 10;

        createIndex(indexName);
        indexRandom(
            true,
            prepareIndex(indexName).setId("1").setSource(rankFeatureField, 0.1, searchField, "A"),
            prepareIndex(indexName).setId("2").setSource(rankFeatureField, 0.2, searchField, "B"),
            prepareIndex(indexName).setId("3").setSource(rankFeatureField, 0.3, searchField, "C"),
            prepareIndex(indexName).setId("4").setSource(rankFeatureField, 0.4, searchField, "D"),
            prepareIndex(indexName).setId("5").setSource(rankFeatureField, 0.5, searchField, "E")
        );

        assertNoFailuresAndResponse(
            prepareSearch().setQuery(
                boolQuery().should(constantScoreQuery(matchQuery(searchField, "A")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "B")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "C")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "D")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "E")).boost(randomFloat()))
            )
                .setRankBuilder(getRankBuilder(rankWindowSize, rankFeatureField))
                .addFetchField(searchField)
                .setTrackTotalHits(true)
                .setAllowPartialSearchResults(true)
                .setSize(2)
                .setFrom(10),
            response -> {
                assertHitCount(response, 5L);
                assertEquals(0, response.getHits().getHits().length);
            }
        );
    }

    public void testRerankerNoMatchingDocs() throws Exception {
        final String indexName = "test_index";
        final String rankFeatureField = "rankFeatureField";
        final String searchField = "searchField";
        final int rankWindowSize = 10;

        createIndex(indexName);
        indexRandom(
            true,
            prepareIndex(indexName).setId("1").setSource(rankFeatureField, 0.1, searchField, "A"),
            prepareIndex(indexName).setId("2").setSource(rankFeatureField, 0.2, searchField, "B"),
            prepareIndex(indexName).setId("3").setSource(rankFeatureField, 0.3, searchField, "C"),
            prepareIndex(indexName).setId("4").setSource(rankFeatureField, 0.4, searchField, "D"),
            prepareIndex(indexName).setId("5").setSource(rankFeatureField, 0.5, searchField, "E")
        );

        assertNoFailuresAndResponse(
            prepareSearch().setQuery(boolQuery().should(constantScoreQuery(matchQuery(searchField, "F")).boost(randomFloat())))
                .setRankBuilder(getRankBuilder(rankWindowSize, rankFeatureField))
                .addFetchField(searchField)
                .setTrackTotalHits(true)
                .setAllowPartialSearchResults(true)
                .setSize(10),
            response -> {
                assertHitCount(response, 0L);
            }
        );
    }

    public void testRerankerNullFieldData() throws Exception {
        final String indexName = "test_index";
        final String rankFeatureField = "rankFeatureField";
        final String searchField = "searchField";
        final int rankWindowSize = 10;

        createIndex(indexName);
        indexRandom(
            true,
            prepareIndex(indexName).setId("1").setSource(searchField, "A"),
            prepareIndex(indexName).setId("2").setSource(searchField, "B"),
            prepareIndex(indexName).setId("3").setSource(rankFeatureField, 0.3, searchField, "C"),
            prepareIndex(indexName).setId("4").setSource(searchField, "D"),
            prepareIndex(indexName).setId("5").setSource(rankFeatureField, 0.5, searchField, "E")
        );

        assertNoFailuresAndResponse(
            prepareSearch().setQuery(
                boolQuery().should(constantScoreQuery(matchQuery(searchField, "A")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "B")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "C")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "D")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "E")).boost(randomFloat()))
            )
                .setRankBuilder(getRankBuilder(rankWindowSize, rankFeatureField))
                .addFetchField(searchField)
                .setTrackTotalHits(true)
                .setAllowPartialSearchResults(true)
                .setSize(10),
            response -> {
                assertHitCount(response, 5L);
                assertEquals(2, response.getHits().getHits().length);
                SearchHit firstHit = response.getHits().getAt(0);
                assertThat(firstHit, hasId("5"));
                assertEquals(firstHit.getScore(), 0.5f, 1e-5f);
                assertThat(firstHit, hasRank(1));
                assertNotNull(firstHit.getFields().get(searchField));

                SearchHit secondHit = response.getHits().getAt(1);
                assertThat(secondHit, hasId("3"));
                assertEquals(secondHit.getScore(), 0.3f, 1e-5f);
                assertThat(secondHit, hasRank(2));
                assertNotNull(secondHit.getFields().get(searchField));
            }
        );
    }

    public void testThrowingRankBuilderAllContextsAreClosedPartialFailures() throws Exception {
        final String indexName = "test_index";
        final String rankFeatureField = "rankFeatureField";
        final String searchField = "searchField";
        final int rankWindowSize = 10;

        // we have less than the max number of nodes here, so not all shards will have failed and "partial" results can be
        // returned
        createIndex(indexName, Settings.builder().put(IndexMetadata.SETTING_NUMBER_OF_SHARDS, 10).build());
        indexRandom(
            true,
            prepareIndex(indexName).setId("1").setSource(rankFeatureField, 0.1, searchField, "A"),
            prepareIndex(indexName).setId("2").setSource(rankFeatureField, 0.2, searchField, "B"),
            prepareIndex(indexName).setId("3").setSource(rankFeatureField, 0.3, searchField, "C"),
            prepareIndex(indexName).setId("4").setSource(rankFeatureField, 0.4, searchField, "D"),
            prepareIndex(indexName).setId("5").setSource(rankFeatureField, 0.5, searchField, "E")
        );

        // we have 10 shards and 2 documents, so when the exception is thrown we know that not all shards will report failures
        assertResponse(
            prepareSearch().setQuery(
                boolQuery().should(constantScoreQuery(matchQuery(searchField, "A")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "B")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "C")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "D")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "E")).boost(randomFloat()))
            )
                .setRankBuilder(getShardThrowingRankBuilder(rankWindowSize, rankFeatureField))
                .addFetchField(searchField)
                .setTrackTotalHits(true)
                .setAllowPartialSearchResults(true)
                .setSize(10),
            response -> {
                assertTrue(response.getFailedShards() > 0);
                assertTrue(
                    Arrays.stream(response.getShardFailures())
                        .allMatch(failure -> failure.getCause().getMessage().equals("This rank builder throws an exception"))
                );
                assertHitCount(response, 5);
                assertTrue(response.getHits().getHits().length == 0);
            }
        );
    }

    public void testThrowingRankBuilderAllContextsAreClosedAllShardsFail() throws Exception {
        final String indexName = "test_index";
        final String rankFeatureField = "rankFeatureField";
        final String searchField = "searchField";
        final int rankWindowSize = 10;

        // we have 1 shard and 2 documents, so when the exception is thrown we know that all shards will have failed
        createIndex(indexName, Settings.builder().put(IndexMetadata.SETTING_NUMBER_OF_SHARDS, 1).build());
        indexRandom(
            true,
            prepareIndex(indexName).setId("1").setSource(rankFeatureField, 0.1, searchField, "A"),
            prepareIndex(indexName).setId("2").setSource(rankFeatureField, 0.2, searchField, "B"),
            prepareIndex(indexName).setId("3").setSource(rankFeatureField, 0.3, searchField, "C"),
            prepareIndex(indexName).setId("4").setSource(rankFeatureField, 0.4, searchField, "D"),
            prepareIndex(indexName).setId("5").setSource(rankFeatureField, 0.5, searchField, "E")
        );

        expectThrows(
            SearchPhaseExecutionException.class,
            () -> prepareSearch().setQuery(
                boolQuery().should(constantScoreQuery(matchQuery(searchField, "A")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "B")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "C")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "D")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "E")).boost(randomFloat()))
            )
                .setRankBuilder(getShardThrowingRankBuilder(rankWindowSize, rankFeatureField))
                .addFetchField(searchField)
                .setTrackTotalHits(true)
                .setAllowPartialSearchResults(true)
                .setSize(10)
                .get()
        );
    }

    public void testRerankFailsOnCoordinator() throws Exception {
        final String indexName = "test_index";
        final String rankFeatureField = "rankFeatureField";
        final String searchField = "searchField";
        final int rankWindowSize = 10;

        createIndex(indexName);
        indexRandom(
            true,
            prepareIndex(indexName).setId("1").setSource(rankFeatureField, 0.1, searchField, "A"),
            prepareIndex(indexName).setId("2").setSource(rankFeatureField, 0.2, searchField, "B"),
            prepareIndex(indexName).setId("3").setSource(rankFeatureField, 0.3, searchField, "C"),
            prepareIndex(indexName).setId("4").setSource(rankFeatureField, 0.4, searchField, "D"),
            prepareIndex(indexName).setId("5").setSource(rankFeatureField, 0.5, searchField, "E")
        );

        expectThrows(
            SearchPhaseExecutionException.class,
            () -> prepareSearch().setQuery(
                boolQuery().should(constantScoreQuery(matchQuery(searchField, "A")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "B")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "C")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "D")).boost(randomFloat()))
                    .should(constantScoreQuery(matchQuery(searchField, "E")).boost(randomFloat()))
            )
                .setRankBuilder(getCoordinatorThrowingRankBuilder(rankWindowSize, rankFeatureField))
                .addFetchField(searchField)
                .setTrackTotalHits(true)
                .setAllowPartialSearchResults(true)
                .setSize(10)
                .get()
        );
    }

    public void testNonBlockingRanking() throws Exception {

    }

    protected abstract RankBuilder getRankBuilder(int rankWindowSize, String rankFeatureField);

    protected abstract RankBuilder getShardThrowingRankBuilder(int rankWindowSize, String rankFeatureField);

    protected abstract RankBuilder getCoordinatorThrowingRankBuilder(int rankWindowSize, String rankFeatureField);

    protected abstract Collection<Class<? extends Plugin>> pluginsNeeded();

    @Override
    protected Collection<Class<? extends Plugin>> nodePlugins() {
        return pluginsNeeded();
    }
}
