/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.rank.textsimilarity;

import org.elasticsearch.common.ParsingException;
import org.elasticsearch.features.NodeFeature;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.QueryRewriteContext;
import org.elasticsearch.license.LicenseUtils;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.search.rank.RankDoc;
import org.elasticsearch.search.retriever.RetrieverBuilder;
import org.elasticsearch.search.retriever.RetrieverParserContext;
import org.elasticsearch.search.retriever.rankdoc.RankDocsQueryBuilder;
import org.elasticsearch.xcontent.ConstructingObjectParser;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.xpack.core.XPackPlugin;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

import static org.elasticsearch.search.rank.RankBuilder.DEFAULT_RANK_WINDOW_SIZE;
import static org.elasticsearch.xcontent.ConstructingObjectParser.constructorArg;
import static org.elasticsearch.xcontent.ConstructingObjectParser.optionalConstructorArg;

/**
 * A {@code RetrieverBuilder} for parsing and constructing a text similarity reranker retriever.
 */
public class TextSimilarityRankRetrieverBuilder extends RetrieverBuilder {

    public static final NodeFeature TEXT_SIMILARITY_RERANKER_RETRIEVER_SUPPORTED = new NodeFeature(
        "text_similarity_reranker_retriever_supported"
    );
    public static final NodeFeature TEXT_SIMILARITY_RERANKER_COMPOSITION_SUPPORTED = new NodeFeature(
        "text_similarity_reranker_retriever_composition_supported"
    );

    public static final ParseField RETRIEVER_FIELD = new ParseField("retriever");
    public static final ParseField INFERENCE_ID_FIELD = new ParseField("inference_id");
    public static final ParseField INFERENCE_TEXT_FIELD = new ParseField("inference_text");
    public static final ParseField FIELD_FIELD = new ParseField("field");
    public static final ParseField RANK_WINDOW_SIZE_FIELD = new ParseField("rank_window_size");

    public static final ConstructingObjectParser<TextSimilarityRankRetrieverBuilder, RetrieverParserContext> PARSER =
        new ConstructingObjectParser<>(TextSimilarityRankBuilder.NAME, args -> {
            RetrieverBuilder retrieverBuilder = (RetrieverBuilder) args[0];
            String inferenceId = (String) args[1];
            String inferenceText = (String) args[2];
            String field = (String) args[3];
            int rankWindowSize = args[4] == null ? DEFAULT_RANK_WINDOW_SIZE : (int) args[4];
            return new TextSimilarityRankRetrieverBuilder(retrieverBuilder, inferenceId, inferenceText, field, rankWindowSize);
        });

    static {
        PARSER.declareNamedObject(constructorArg(), (p, c, n) -> {
            RetrieverBuilder innerRetriever = p.namedObject(RetrieverBuilder.class, n, c);
            c.trackRetrieverUsage(innerRetriever.getName());
            return innerRetriever;
        }, RETRIEVER_FIELD);
        PARSER.declareString(constructorArg(), INFERENCE_ID_FIELD);
        PARSER.declareString(constructorArg(), INFERENCE_TEXT_FIELD);
        PARSER.declareString(constructorArg(), FIELD_FIELD);
        PARSER.declareInt(optionalConstructorArg(), RANK_WINDOW_SIZE_FIELD);

        RetrieverBuilder.declareBaseParserFields(TextSimilarityRankBuilder.NAME, PARSER);
    }

    public static TextSimilarityRankRetrieverBuilder fromXContent(XContentParser parser, RetrieverParserContext context)
        throws IOException {
        if (context.clusterSupportsFeature(TEXT_SIMILARITY_RERANKER_RETRIEVER_SUPPORTED) == false) {
            throw new ParsingException(parser.getTokenLocation(), "unknown retriever [" + TextSimilarityRankBuilder.NAME + "]");
        }
        if (context.clusterSupportsFeature(TEXT_SIMILARITY_RERANKER_COMPOSITION_SUPPORTED) == false) {
            throw new UnsupportedOperationException(
                "[text_similarity_reranker] retriever composition feature is not supported by all nodes in the cluster"
            );
        }
        if (TextSimilarityRankBuilder.TEXT_SIMILARITY_RERANKER_FEATURE.check(XPackPlugin.getSharedLicenseState()) == false) {
            throw LicenseUtils.newComplianceException(TextSimilarityRankBuilder.NAME);
        }
        return PARSER.apply(parser, context);
    }

    private final RetrieverBuilder innerRetriever;
    private final String inferenceId;
    private final String inferenceText;
    private final String field;
    private final int rankWindowSize;

    public TextSimilarityRankRetrieverBuilder(
        RetrieverBuilder retrieverBuilder,
        String inferenceId,
        String inferenceText,
        String field,
        int rankWindowSize
    ) {
        this.innerRetriever = retrieverBuilder;
        this.inferenceId = inferenceId;
        this.inferenceText = inferenceText;
        this.field = field;
        this.rankWindowSize = rankWindowSize;
    }

    public TextSimilarityRankRetrieverBuilder(
        RetrieverBuilder innerRetriever,
        String inferenceId,
        String inferenceText,
        String field,
        int rankWindowSize,
        Float minScore,
        String retrieverName,
        List<QueryBuilder> preFilterQueryBuilders
    ) {
        this.innerRetriever = innerRetriever;
        this.inferenceId = inferenceId;
        this.inferenceText = inferenceText;
        this.field = field;
        this.minScore = minScore;
        this.retrieverName = retrieverName;
        this.preFilterQueryBuilders = preFilterQueryBuilders;
        this.rankWindowSize = rankWindowSize;
    }

    private TextSimilarityRankRetrieverBuilder clone(RetrieverBuilder rewrittenInnerRetriever, List<QueryBuilder> newPreFilters) {
        return new TextSimilarityRankRetrieverBuilder(
            rewrittenInnerRetriever,
            inferenceId,
            inferenceText,
            field,
            rankWindowSize,
            minScore,
            retrieverName,
            newPreFilters
        );
    }

    @Override
    public final RetrieverBuilder rewrite(QueryRewriteContext ctx) throws IOException {
        // Rewrite prefilters
        boolean hasChanged = false;
        var newPreFilters = rewritePreFilters(ctx);
        hasChanged |= newPreFilters != preFilterQueryBuilders;

        // Rewrite retriever sources
        RetrieverBuilder rewrittenInnerRetriever = innerRetriever.rewrite(ctx);
        hasChanged |= rewrittenInnerRetriever != innerRetriever;
        if (hasChanged) {
            return clone(rewrittenInnerRetriever, newPreFilters);
        }
        return this;
    }

    @Override
    public QueryBuilder topDocsQuery() {
        return innerRetriever.topDocsQuery();
    }

    @Override
    public QueryBuilder explainQuery() {
        assert rankDocs != null;
        TextSimilarityRankDoc[] mappedRankDocs = new TextSimilarityRankDoc[rankDocs.length];
        for (int i = 0; i < rankDocs.length; i++) {
            RankDoc rd = rankDocs[i];
            mappedRankDocs[i] = new TextSimilarityRankDoc(rd.doc, rd.score, rd.shardIndex, inferenceId, field);
        }
        return new RankDocsQueryBuilder(mappedRankDocs, new QueryBuilder[]{innerRetriever.explainQuery()}, true);
    }

    @Override
    public void extractToSearchSourceBuilder(SearchSourceBuilder searchSourceBuilder, boolean compoundUsed) {
        if (preFilterQueryBuilders.isEmpty() == false) {
            innerRetriever.getPreFilterQueryBuilders().addAll(preFilterQueryBuilders);
        }
        innerRetriever.extractToSearchSourceBuilder(searchSourceBuilder, compoundUsed);
        searchSourceBuilder.rankBuilder(
            new TextSimilarityRankBuilder(this.field, this.inferenceId, this.inferenceText, this.rankWindowSize, this.minScore)
        );
    }

    @Override
    public boolean isCompound(){
        return innerRetriever.isCompound();
    }

    @Override
    public String getName() {
        return TextSimilarityRankBuilder.NAME;
    }

    public int rankWindowSize() {
        return rankWindowSize;
    }

    @Override
    protected void doToXContent(XContentBuilder builder, Params params) throws IOException {
        builder.field(RETRIEVER_FIELD.getPreferredName(), innerRetriever);
        builder.field(INFERENCE_ID_FIELD.getPreferredName(), inferenceId);
        builder.field(INFERENCE_TEXT_FIELD.getPreferredName(), inferenceText);
        builder.field(FIELD_FIELD.getPreferredName(), field);
        builder.field(RANK_WINDOW_SIZE_FIELD.getPreferredName(), rankWindowSize);
    }

    @Override
    public boolean doEquals(Object other) {
        TextSimilarityRankRetrieverBuilder that = (TextSimilarityRankRetrieverBuilder) other;
        return Objects.equals(inferenceId, that.innerRetriever)
            && Objects.equals(inferenceId, that.inferenceId)
            && Objects.equals(inferenceText, that.inferenceText)
            && Objects.equals(field, that.field)
            && Objects.equals(rankWindowSize, that.rankWindowSize)
            && Objects.equals(minScore, that.minScore);
    }

    @Override
    public int doHashCode() {
        return Objects.hash(inferenceId, inferenceText, field, rankWindowSize, minScore);
    }
}
