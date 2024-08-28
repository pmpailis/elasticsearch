/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.retriever;

import org.elasticsearch.index.query.MatchAllQueryBuilder;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.QueryRewriteContext;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.test.AbstractMultiClustersTestCase;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.List;


public class MinimalCompoundRetrieverIT extends AbstractMultiClustersTestCase {


    public static class CompoundRetriever extends RetrieverBuilder {

        private final List<RetrieverBuilder> sources;

        private CompoundRetriever(List<RetrieverBuilder> sources) {
            this.sources = sources;
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
            if(sources.isEmpty()){
                StandardRetrieverBuilder standardRetrieverBuilder = new StandardRetrieverBuilder();
                standardRetrieverBuilder.queryBuilder = new MatchAllQueryBuilder();
                return standardRetrieverBuilder;
            }
            return sources.get(0);
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
            // no-op
        }

        @Override
        protected boolean doEquals(Object o) {
            return false;
        }

        @Override
        protected int doHashCode() {
            return 0;
        }
    }
}
