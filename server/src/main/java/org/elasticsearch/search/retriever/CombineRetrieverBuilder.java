/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.retriever;

import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.util.SetOnce;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.search.MultiSearchRequest;
import org.elasticsearch.action.search.MultiSearchResponse;
import org.elasticsearch.action.search.SearchRequest;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.action.search.TransportMultiSearchAction;
import org.elasticsearch.index.query.BoolQueryBuilder;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.QueryRewriteContext;
import org.elasticsearch.search.builder.PointInTimeBuilder;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.search.fetch.StoredFieldsContext;
import org.elasticsearch.search.rank.RankDoc;
import org.elasticsearch.search.sort.FieldSortBuilder;
import org.elasticsearch.search.sort.ScoreSortBuilder;
import org.elasticsearch.search.sort.ShardDocSortField;
import org.elasticsearch.search.sort.SortBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * This abstract retriever is a compound retriever.
 * It has a set of child retrievers, each returning a set of top documents.
 * These documents are then combined and ranked according to the implementation of {@code combineQueryPhaseResults}.
 */
public abstract class CombineRetrieverBuilder<T extends CombineRetrieverBuilder<T>> extends RetrieverBuilder {
    public record RetrieverSource(RetrieverBuilder retriever, SearchSourceBuilder source) {}

    protected final int windowSize;
    protected final List<RetrieverSource> childRetrievers;

    protected CombineRetrieverBuilder(List<RetrieverSource> childRetrievers, int windowSize) {
        this.windowSize = windowSize;
        this.childRetrievers = childRetrievers;
    }

    @SuppressWarnings("unchecked")
    public T addChild(RetrieverBuilder retrieverBuilder) {
        childRetrievers.add(new RetrieverSource(retrieverBuilder, null));
        return (T) this;
    }

    /**
     * Returns a clone of the original retriever, replacing the sub-retrievers with
     * the provided {@code newChildRetrievers}.
     */
    public abstract T clone(T original, List<RetrieverSource> newChildRetrievers);

    /**
     * Combines the provided {@code rankResults} to return the final top documents.
     */
    public abstract RankDoc[] combineQueryPhaseResults(List<ScoreDoc[]> rankResults);

    @Override
    public final boolean isCompound() {
        return true;
    }

    @Override
    @SuppressWarnings("unchecked")
    public final RetrieverBuilder rewrite(QueryRewriteContext ctx) throws IOException {
        if (ctx.pointInTimeBuilder() == null) {
            throw new IllegalStateException("PIT is required");
        }

        // Rewrite prefilters
        boolean hasChanged = false;
        var newPreFilters = rewritePreFilters(ctx);
        hasChanged |= newPreFilters != preFilterQueryBuilders;

        // Rewrite retriever sources
        List<RetrieverSource> newRetrievers = new ArrayList<>();
        for (var entry : childRetrievers) {
            RetrieverBuilder newRetriever = entry.retriever.rewrite(ctx);
            if (newRetriever != entry.retriever) {
                newRetrievers.add(new RetrieverSource(newRetriever, null));
                hasChanged |= newRetriever != entry.retriever;
            } else if (newRetriever == entry.retriever) {
                var sourceBuilder = entry.source != null ? entry.source : createSearchSourceBuilder(ctx.pointInTimeBuilder(), newRetriever);
                var rewrittenSource = sourceBuilder.rewrite(ctx);
                newRetrievers.add(new RetrieverSource(newRetriever, rewrittenSource));
                hasChanged |= rewrittenSource != entry.source;
            }
        }
        if (hasChanged) {
            return clone((T) this, newRetrievers);
        }

        // execute searches
        final SetOnce<RankDoc[]> results = new SetOnce<>();
        final MultiSearchRequest multiSearchRequest = new MultiSearchRequest();
        for (var entry : childRetrievers) {
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
                        topDocs.add(getTopDocs(item.getResponse()));
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

        return new RankDocsRetrieverBuilder(windowSize, newRetrievers.stream().map(s -> s.retriever).toList(), results::get, newPreFilters);
    }

    @Override
    public final QueryBuilder topDocsQuery(QueryBuilder leadQuery) {
        throw new IllegalStateException(getName() + " cannot be nested");
    }

    @Override
    public final void extractToSearchSourceBuilder(SearchSourceBuilder searchSourceBuilder, boolean compoundUsed) {
        throw new IllegalStateException("Should not be called, missing a rewrite?");
    }

    @Override
    @SuppressWarnings("unchecked")
    public boolean doEquals(Object o) {
        CombineRetrieverBuilder<?> that = (CombineRetrieverBuilder<?>) o;
        return windowSize == that.windowSize && Objects.equals(childRetrievers, that.childRetrievers);
    }

    @Override
    public int doHashCode() {
        return Objects.hash(childRetrievers, windowSize);
    }

    private SearchSourceBuilder createSearchSourceBuilder(PointInTimeBuilder pit, RetrieverBuilder retrieverBuilder) {
        var sourceBuilder = new SearchSourceBuilder().pointInTimeBuilder(pit)
            .trackTotalHits(false)
            .storedFields(new StoredFieldsContext(false))
            .size(windowSize);
        retrieverBuilder.extractToSearchSourceBuilder(sourceBuilder, false);

        // apply the pre-filters
        if (preFilterQueryBuilders.size() > 0) {
            QueryBuilder query = sourceBuilder.query();
            BoolQueryBuilder newQuery = new BoolQueryBuilder();
            if (query != null) {
                newQuery.must(query);
            }
            preFilterQueryBuilders.stream().forEach(newQuery::filter);
            sourceBuilder.query(newQuery);
        }

        // Record the shard id in the sort result
        List<SortBuilder<?>> sortBuilders = sourceBuilder.sorts() != null ? new ArrayList<>(sourceBuilder.sorts()) : new ArrayList<>();
        if (sortBuilders.isEmpty()) {
            sortBuilders.add(new ScoreSortBuilder());
        }
        sortBuilders.add(new FieldSortBuilder(FieldSortBuilder.SHARD_DOC_FIELD_NAME));
        sourceBuilder.sort(sortBuilders);
        return sourceBuilder;
    }

    private ScoreDoc[] getTopDocs(SearchResponse searchResponse) {
        int size = Math.min(windowSize, searchResponse.getHits().getHits().length);
        ScoreDoc[] docs = new ScoreDoc[size];
        for (int i = 0; i < size; i++) {
            var hit = searchResponse.getHits().getAt(i);
            long sortValue = (long) hit.getRawSortValues()[hit.getRawSortValues().length - 1];
            int doc = ShardDocSortField.decodeDoc(sortValue);
            int shardRequestIndex = ShardDocSortField.decodeShardRequestIndex(sortValue);
            docs[i] = new ScoreDoc(doc, hit.getScore(), shardRequestIndex);
        }
        return docs;
    }
}
