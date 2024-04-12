/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.retriever;

import org.elasticsearch.common.ParsingException;
import org.elasticsearch.features.NodeFeature;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.search.rank.rerank.RerankingRankBuilder;
import org.elasticsearch.xcontent.ObjectParser;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * A standard retriever is used to represent anything that is a query along
 * with some elements to specify parameters for that query.
 */
public final class RandomRankRetrieverBuilder extends RetrieverBuilder {

    public static final String NAME = "random";
    public static final NodeFeature RANDOM_RETRIEVER_SUPPORTED = new NodeFeature("random_retriever_supported");

    public static final ParseField RETRIEVERS_FIELD = new ParseField("retrievers");
    public static final ParseField WINDOW_SIZE_FIELD = new ParseField("window_size");

    public static final ObjectParser<RandomRankRetrieverBuilder, RetrieverParserContext> PARSER = new ObjectParser<>(
        NAME,
        RandomRankRetrieverBuilder::new
    );

    static {
        PARSER.declareObjectArray((r, v) -> r.retrieverBuilders = v, (p, c) -> {
            p.nextToken();
            String name = p.currentName();
            RetrieverBuilder retrieverBuilder = p.namedObject(RetrieverBuilder.class, name, c);
            p.nextToken();
            return retrieverBuilder;
        }, RETRIEVERS_FIELD);
        PARSER.declareInt((r, v) -> r.windowSize = v, WINDOW_SIZE_FIELD);

        RetrieverBuilder.declareBaseParserFields(NAME, PARSER);
}

    public static RandomRankRetrieverBuilder fromXContent(XContentParser parser, RetrieverParserContext context) throws IOException {
        if (context.clusterSupportsFeature(RANDOM_RETRIEVER_SUPPORTED) == false) {
            throw new ParsingException(parser.getTokenLocation(), "unknown retriever [" + NAME + "]");
        }
        return PARSER.apply(parser, context);
    }
    List<RetrieverBuilder> retrieverBuilders = Collections.emptyList();
    int windowSize = 100;

    @Override
    public void extractToSearchSourceBuilder(SearchSourceBuilder searchSourceBuilder, boolean compoundUsed) {
        for (RetrieverBuilder retrieverBuilder : retrieverBuilders) {
            if (preFilterQueryBuilders.isEmpty() == false) {
                retrieverBuilder.getPreFilterQueryBuilders().addAll(preFilterQueryBuilders);
            }

            retrieverBuilder.extractToSearchSourceBuilder(searchSourceBuilder, true);
        }

        searchSourceBuilder.rankBuilder(new RerankingRankBuilder(windowSize, "body"));
    }

    // ---- FOR TESTING XCONTENT PARSING ----

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public void doToXContent(XContentBuilder builder, Params params) throws IOException {
        if (retrieverBuilders.isEmpty() == false) {
            builder.startArray(RETRIEVERS_FIELD.getPreferredName());

            for (RetrieverBuilder retrieverBuilder : retrieverBuilders) {
                builder.startObject();
                builder.field(retrieverBuilder.getName());
                retrieverBuilder.toXContent(builder, params);
                builder.endObject();
            }

            builder.endArray();
        }

        builder.field(WINDOW_SIZE_FIELD.getPreferredName(), windowSize);
    }

    @Override
    public boolean doEquals(Object o) {
        RandomRankRetrieverBuilder that = (RandomRankRetrieverBuilder) o;
        return windowSize == that.windowSize
            && Objects.equals(retrieverBuilders, that.retrieverBuilders);
    }

    @Override
    public int doHashCode() {
        return Objects.hash(retrieverBuilders, windowSize);
    }

    // ---- END FOR TESTING ----
}
