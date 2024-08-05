/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.retriever.rankdoc;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.index.query.BoolQueryBuilder;
import org.elasticsearch.index.query.DisMaxQueryBuilder;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.QueryRewriteContext;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.search.rank.RankDoc;
import org.elasticsearch.search.retriever.RetrieverBuilder;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * An {@link RetrieverBuilder} that is used to
 */
public class RankDocsRetrieverBuilder extends RetrieverBuilder {
    private static final Logger logger = LogManager.getLogger(RankDocsRetrieverBuilder.class);

    public static final String NAME = "rank_docs";
    private final int windowSize;
    private final List<RetrieverBuilder> sources;
    private final Supplier<RankDoc[]> rankDocs;

    public RankDocsRetrieverBuilder(
        int windowSize,
        List<RetrieverBuilder> rewritten,
        Supplier<RankDoc[]> rankDocs,
        List<QueryBuilder> preFilterQueryBuilders
    ) {
        this.windowSize = windowSize;
        this.rankDocs = rankDocs;
        this.sources = rewritten;
        this.preFilterQueryBuilders = preFilterQueryBuilders;
    }

    @Override
    public String getName() {
        return NAME;
    }

    private boolean sourceShouldRewrite(QueryRewriteContext ctx) throws IOException {
        for (var source : sources) {
            var newSource = source.rewrite(ctx);
            if (newSource != source) {
                return true;
            }
        }
        return false;
    }

    @Override
    public RetrieverBuilder rewrite(QueryRewriteContext ctx) throws IOException {
        assert sourceShouldRewrite(ctx) == false : "Retriever sources should be rewritten first";
        var rewrittenFilters = rewritePreFilters(ctx);
        if (rewrittenFilters != preFilterQueryBuilders) {
            return new RankDocsRetrieverBuilder(windowSize, sources, rankDocs, rewrittenFilters);
        }
        return this;
    }

    @Override
    public QueryBuilder topDocsQuery() {
        DisMaxQueryBuilder disMax = new DisMaxQueryBuilder().tieBreaker(0f);
        for (var source : sources) {
            var query = source.topDocsQuery();
            if (query != null) {
                if (source.retrieverName() != null) {
                    query.queryName(source.retrieverName());
                }
                disMax.add(query);
            }
        }
        // ignore prefilters of this level, they are already propagated to children
        return disMax;
    }

    @Override
    public void extractToSearchSourceBuilder(SearchSourceBuilder searchSourceBuilder, boolean compoundUsed) {
        searchSourceBuilder.sort(Collections.singletonList(new RankDocsSortBuilder(rankDocs.get())));
        if (searchSourceBuilder.explain() != null && searchSourceBuilder.explain()) {
            searchSourceBuilder.trackScores(true);
        }
        var bq = new BoolQueryBuilder();
        var rankQuery = new RankDocsQueryBuilder(rankDocs.get());
        if (searchSourceBuilder.aggregations() != null) {
            bq.should(rankQuery);
            searchSourceBuilder.postFilter(rankQuery);
        } else {
            bq.must(rankQuery);
        }
        for (var preFilterQueryBuilder : preFilterQueryBuilders) {
            bq.filter(preFilterQueryBuilder);
        }
        QueryBuilder originalQuery = topDocsQuery();
        if (originalQuery != null) {
            bq.should(originalQuery);
        }
        searchSourceBuilder.query(bq);
    }

    @Override
    protected boolean doEquals(Object o) {
        RankDocsRetrieverBuilder other = (RankDocsRetrieverBuilder) o;
        return Arrays.equals(rankDocs.get(), other.rankDocs.get()) && sources.equals(other.sources);
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(super.hashCode(), Arrays.hashCode(rankDocs.get()), windowSize);
    }

    @Override
    protected void doToXContent(XContentBuilder builder, Params params) throws IOException {
        throw new UnsupportedOperationException("Not supported");
    }
}
