/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.retriever;

import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.util.SetOnce;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.search.MultiSearchRequest;
import org.elasticsearch.action.search.MultiSearchResponse;
import org.elasticsearch.action.search.SearchRequest;
import org.elasticsearch.action.search.SearchRequestBuilder;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.action.search.TransportMultiSearchAction;
import org.elasticsearch.common.util.Maps;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.index.query.QueryRewriteContext;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.search.MockSearchService;
import org.elasticsearch.search.builder.PointInTimeBuilder;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.search.rank.RankDoc;
import org.elasticsearch.search.rank.TestRankDoc;
import org.elasticsearch.search.retriever.rankdoc.RankDocsQueryBuilderTests;
import org.elasticsearch.search.sort.FieldSortBuilder;
import org.elasticsearch.search.sort.ScoreSortBuilder;
import org.elasticsearch.search.sort.SortBuilder;
import org.elasticsearch.test.ESIntegTestCase;
import org.elasticsearch.test.hamcrest.ElasticsearchAssertions;
import org.elasticsearch.xcontent.XContentBuilder;
import org.junit.Before;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.hamcrest.Matchers.equalTo;

public class RankDocRetrieverBuilderIT extends ESIntegTestCase {

    @Override
    protected Collection<Class<? extends Plugin>> nodePlugins() {
        return List.of(MockSearchService.TestPlugin.class, RankDocsQueryBuilderTests.RankDocsPlugin.class);
    }

    public record RetrieverSource(RetrieverBuilder retriever, SearchSourceBuilder source) {}

    private static String INDEX_DOCS = "docs";
    private static String INDEX_QUERIES = "queries";
    private static final String ID_FIELD = "_id";
    private static final String QUERY_FIELD = "query";

    @Before
    public void setup() throws Exception {
        createIndex(INDEX_DOCS);
        index(INDEX_DOCS, "doc_0", "{}");
        index(INDEX_DOCS, "doc_1", "{}");
        index(INDEX_DOCS, "doc_2", "{}");
        refresh(INDEX_DOCS);

        createIndex(INDEX_QUERIES);
        index(INDEX_QUERIES, "query_0", "{ \"" + QUERY_FIELD + "\": \"doc_2\"}");
        index(INDEX_QUERIES, "query_1", "{ \"" + QUERY_FIELD + "\": \"doc_1\"}");
        index(INDEX_QUERIES, "query_2", "{ \"" + QUERY_FIELD + "\": \"doc_0\"}");
        refresh(INDEX_QUERIES);
    }

    public void testRankDocsRetriever() {
        final int rankWindowSize = 100;
        SearchSourceBuilder source = new SearchSourceBuilder();
        StandardRetrieverBuilder standard0 = new StandardRetrieverBuilder();
        standard0.queryBuilder = QueryBuilders.termQuery(ID_FIELD, "doc_0");
        StandardRetrieverBuilder standard1 = new StandardRetrieverBuilder();
        standard1.queryBuilder = QueryBuilders.termQuery(ID_FIELD, "doc_1");
        source.retriever(
            new CompoundRetrieverWithRankDocs(
                rankWindowSize,
                Arrays.asList(new RetrieverSource(standard0, null), new RetrieverSource(standard1, null))
            )
        );
        SearchRequestBuilder req = client().prepareSearch(INDEX_DOCS, INDEX_QUERIES).setSource(source);
        ElasticsearchAssertions.assertResponse(req, resp -> {
            assertNull(resp.pointInTimeId());
            assertNotNull(resp.getHits().getTotalHits());
            assertThat(resp.getHits().getTotalHits().value, equalTo(2L));
            assertThat(resp.getHits().getTotalHits().relation, equalTo(TotalHits.Relation.EQUAL_TO));
            assertThat(resp.getHits().getAt(0).getId(), equalTo("doc_1"));
            assertThat(resp.getHits().getAt(1).getId(), equalTo("doc_0"));
        });
    }

    class CompoundRetrieverWithRankDocs extends RetrieverBuilder {

        private final List<RetrieverSource> sources;
        private final int rankWindowSize;

        private CompoundRetrieverWithRankDocs(int rankWindowSize, List<RetrieverSource> sources) {
            this.rankWindowSize = rankWindowSize;
            this.sources = Collections.unmodifiableList(sources);
        }

        @Override
        public boolean isCompound() {
            return true;
        }

        @Override
        public QueryBuilder topDocsQuery() {
            throw new UnsupportedOperationException("should not be called");
        }

        @Override
        public RetrieverBuilder rewrite(QueryRewriteContext ctx) throws IOException {
            if (ctx.getPointInTimeBuilder() == null) {
                throw new IllegalStateException("PIT is required");
            }

            // Rewrite prefilters
            boolean hasChanged = false;
            var newPreFilters = rewritePreFilters(ctx);
            hasChanged |= newPreFilters != preFilterQueryBuilders;

            // Rewrite retriever sources
            List<RetrieverSource> newRetrievers = new ArrayList<>();
            for (var entry : sources) {
                RetrieverBuilder newRetriever = entry.retriever.rewrite(ctx);
                if (newRetriever != entry.retriever) {
                    newRetrievers.add(new RetrieverSource(newRetriever, null));
                    hasChanged |= newRetriever != entry.retriever;
                } else if (newRetriever == entry.retriever) {
                    var sourceBuilder = entry.source != null
                        ? entry.source
                        : createSearchSourceBuilder(ctx.getPointInTimeBuilder(), newRetriever);
                    var rewrittenSource = sourceBuilder.rewrite(ctx);
                    newRetrievers.add(new RetrieverSource(newRetriever, rewrittenSource));
                    hasChanged |= rewrittenSource != entry.source;
                }
            }
            if (hasChanged) {
                return new CompoundRetrieverWithRankDocs(rankWindowSize, newRetrievers);
            }

            // execute searches
            final SetOnce<RankDoc[]> results = new SetOnce<>();
            final MultiSearchRequest multiSearchRequest = new MultiSearchRequest();
            for (var entry : sources) {
                SearchRequest searchRequest = new SearchRequest().source(entry.source);
                // The can match phase can reorder shards, so we disable it to ensure the stable ordering
                searchRequest.setPreFilterShardSize(Integer.MAX_VALUE);
                multiSearchRequest.add(searchRequest);
            }
            ctx.registerAsyncAction((client, listener) -> {
                client.execute(TransportMultiSearchAction.TYPE, multiSearchRequest, new ActionListener<>() {
                    @Override
                    public void onResponse(MultiSearchResponse items) {
                        List<ScoreDoc[]> topDocs = new ArrayList<>();
                        for (int i = 0; i < items.getResponses().length; i++) {
                            var item = items.getResponses()[i];
                            var rankDocs = getRankDocs(item.getResponse());
                            sources.get(i).retriever().setRankDocs(rankDocs);
                            topDocs.add(rankDocs);
                        }
                        results.set(combineQueryPhaseResults(topDocs));
                        listener.onResponse(null);
                    }

                    @Override
                    public void onFailure(Exception e) {
                        listener.onFailure(e);
                    }
                });
            });

            return new RankDocsRetrieverBuilder(
                rankWindowSize,
                newRetrievers.stream().map(s -> s.retriever).toList(),
                results::get,
                newPreFilters
            );
        }

        @Override
        public void extractToSearchSourceBuilder(SearchSourceBuilder searchSourceBuilder, boolean compoundUsed) {
            throw new UnsupportedOperationException("should not be called");
        }

        @Override
        public String getName() {
            return "compound_retriever";
        }

        @Override
        protected void doToXContent(XContentBuilder builder, Params params) throws IOException {

        }

        @Override
        protected boolean doEquals(Object o) {
            return false;
        }

        @Override
        protected int doHashCode() {
            return 0;
        }

        private RankDoc[] getRankDocs(SearchResponse searchResponse) {
            int size = Math.min(rankWindowSize, searchResponse.getHits().getHits().length);
            RankDoc[] docs = new RankDoc[size];
            for (int i = 0; i < size; i++) {
                var hit = searchResponse.getHits().getAt(i);
                long sortValue = (long) hit.getRawSortValues()[hit.getRawSortValues().length - 1];
                int doc = decodeDoc(sortValue);
                int shardRequestIndex = decodeShardRequestIndex(sortValue);
                docs[i] = new TestRankDoc(doc, hit.getScore(), shardRequestIndex);
                docs[i].rank = i;
            }
            return docs;
        }

        public static int decodeDoc(long value) {
            return (int) value;
        }

        public static int decodeShardRequestIndex(long value) {
            return (int) (value >> 32);
        }

        /**
         * Combines the provided {@code rankResults} to return the final top documents.
         */
        public RankDoc[] combineQueryPhaseResults(List<ScoreDoc[]> rankResults) {
            Map<RankDoc.RankKey, TestRankDoc> docsToRankResults = Maps.newMapWithExpectedSize(rankWindowSize);
            for (var rrfRankResult : rankResults) {
                for (ScoreDoc scoreDoc : rrfRankResult) {
                    docsToRankResults.compute(new RankDoc.RankKey(scoreDoc.doc, scoreDoc.shardIndex), (key, value) -> {
                        if (value == null) {
                            value = new TestRankDoc(scoreDoc.doc, scoreDoc.score, scoreDoc.shardIndex);
                        }
                        value.score = Math.max(value.score, scoreDoc.score);
                        return value;
                    });
                }
            }
            // sort the results based on rrf score, tiebreaker based on smaller doc id
            TestRankDoc[] sortedResults = docsToRankResults.values().toArray(TestRankDoc[]::new);
            Arrays.sort(sortedResults, (TestRankDoc rrf1, TestRankDoc rrf2) -> {
                if (rrf1.score != rrf2.score) {
                    return rrf1.score < rrf2.score ? 1 : -1;
                }
                return rrf1.doc < rrf2.doc ? -1 : 1;
            });
            // trim the results if needed, otherwise each shard will always return `rank_window_size` results.
            // pagination and all else will happen on the coordinator when combining the shard responses
            TestRankDoc[] topResults = new TestRankDoc[Math.min(rankWindowSize, sortedResults.length)];
            for (int rank = 0; rank < topResults.length; ++rank) {
                topResults[rank] = sortedResults[rank];
                topResults[rank].rank = rank + 1;
            }
            return topResults;
        }
    }

    private SearchSourceBuilder createSearchSourceBuilder(PointInTimeBuilder pit, RetrieverBuilder retrieverBuilder) {
        var sourceBuilder = new SearchSourceBuilder().pointInTimeBuilder(pit).trackTotalHits(false).size(100);
        retrieverBuilder.extractToSearchSourceBuilder(sourceBuilder, false);

        // Record the shard id in the sort result
        List<SortBuilder<?>> sortBuilders = sourceBuilder.sorts() != null ? new ArrayList<>(sourceBuilder.sorts()) : new ArrayList<>();
        if (sortBuilders.isEmpty()) {
            sortBuilders.add(new ScoreSortBuilder());
        }
        sortBuilders.add(new FieldSortBuilder(FieldSortBuilder.SHARD_DOC_FIELD_NAME));
        sourceBuilder.sort(sortBuilders);
        return sourceBuilder;
    }
}
