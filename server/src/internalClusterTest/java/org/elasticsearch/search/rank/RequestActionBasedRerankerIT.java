/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank;

import org.apache.lucene.search.Query;
import org.elasticsearch.TransportVersion;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.ActionRequest;
import org.elasticsearch.action.ActionRequestValidationException;
import org.elasticsearch.action.ActionResponse;
import org.elasticsearch.action.ActionType;
import org.elasticsearch.action.support.ActionFilters;
import org.elasticsearch.action.support.HandledTransportAction;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.common.inject.Inject;
import org.elasticsearch.common.io.stream.NamedWriteableRegistry;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.util.concurrent.EsExecutors;
import org.elasticsearch.plugins.ActionPlugin;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.plugins.SearchPlugin;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.rank.context.QueryPhaseRankCoordinatorContext;
import org.elasticsearch.search.rank.context.QueryPhaseRankShardContext;
import org.elasticsearch.search.rank.context.RankFeaturePhaseRankCoordinatorContext;
import org.elasticsearch.search.rank.context.RankFeaturePhaseRankShardContext;
import org.elasticsearch.search.rank.feature.RankFeatureShardResult;
import org.elasticsearch.search.rank.request.RequestRankFeaturePhaseRankCoordinatorContext;
import org.elasticsearch.search.rank.rerank.RerankingQueryPhaseRankCoordinatorContext;
import org.elasticsearch.search.rank.rerank.RerankingQueryPhaseRankShardContext;
import org.elasticsearch.search.rank.rerank.RerankingRankFeaturePhaseRankShardContext;
import org.elasticsearch.tasks.Task;
import org.elasticsearch.test.ESIntegTestCase;
import org.elasticsearch.transport.TransportService;
import org.elasticsearch.xcontent.ConstructingObjectParser;
import org.elasticsearch.xcontent.NamedXContentRegistry;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;

import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Objects;

import static org.elasticsearch.index.query.QueryBuilders.boolQuery;
import static org.elasticsearch.index.query.QueryBuilders.constantScoreQuery;
import static org.elasticsearch.index.query.QueryBuilders.matchQuery;
import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertHitCount;
import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertNoFailuresAndResponse;
import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.hasId;
import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.hasRank;
import static org.elasticsearch.xcontent.ConstructingObjectParser.constructorArg;
import static org.elasticsearch.xcontent.ConstructingObjectParser.optionalConstructorArg;

@ESIntegTestCase.ClusterScope(minNumDataNodes = 3)
public class RequestActionBasedRerankerIT extends ESIntegTestCase {

    private static final TestRerankingActionType TEST_RERANKING_ACTION_TYPE = new TestRerankingActionType("internal:test_reranking_action");

    public void testRequestBasedReranker() throws Exception {
        final String indexName = "test_index";
        final String rankFeatureField = "rankFeatureField";
        final String searchField = "searchField";
        final String inferenceId = "inferenceId";
        final String inferenceText = "some query text";
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
                .setRankBuilder(new RerankerBasedRankBuilder(rankWindowSize, rankFeatureField, inferenceId, inferenceText))
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

    public void testExternalServiceThrowsAnException() throws Exception {

    }

    public void testUnknownExternalService() throws Exception {

    }

    public void testPaginatingResults() throws Exception {

    }

    public static class RerankerServicePlugin extends Plugin implements ActionPlugin {

        @Override
        public Collection<ActionHandler<? extends ActionRequest, ? extends ActionResponse>> getActions() {
            return List.of(new ActionHandler<>(TEST_RERANKING_ACTION_TYPE, TestRerankingTransportAction.class));
        }
    }

    public static class InferenceBasedRerankerPlugin extends Plugin implements SearchPlugin {

        private static final String INFERENCE_BASED_RANK_BUILDER_NAME = "inference-based-rank";

        @Override
        public List<NamedWriteableRegistry.Entry> getNamedWriteables() {
            return List.of(
                new NamedWriteableRegistry.Entry(RankBuilder.class, INFERENCE_BASED_RANK_BUILDER_NAME, RerankerBasedRankBuilder::new),
                new NamedWriteableRegistry.Entry(RankShardResult.class, "rank-feature-shard", RankFeatureShardResult::new)
            );
        }

        @Override
        public List<NamedXContentRegistry.Entry> getNamedXContent() {
            return List.of(
                new NamedXContentRegistry.Entry(
                    RankBuilder.class,
                    new ParseField(INFERENCE_BASED_RANK_BUILDER_NAME),
                    RerankerBasedRankBuilder::fromXContent
                )
            );
        }
    }

    @Override
    protected Collection<Class<? extends Plugin>> nodePlugins() {
        return List.of(RerankerServicePlugin.class, InferenceBasedRerankerPlugin.class);
    }

    public static class TestRerankingActionType extends ActionType<TestRerankingActionResponse> {
        TestRerankingActionType(String name) {
            super(name);
        }
    }

    public static class TestRerankingActionRequest extends ActionRequest {

        private final List<String> docFeatures;

        public TestRerankingActionRequest(List<String> docFeatures) {
            super();
            this.docFeatures = docFeatures;
        }

        public TestRerankingActionRequest(StreamInput in) throws IOException {
            super(in);
            this.docFeatures = in.readCollectionAsList(StreamInput::readString);
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            super.writeTo(out);
            out.writeCollection(docFeatures, StreamOutput::writeString);
        }

        @Override
        public ActionRequestValidationException validate() {
            return null;
        }
    }

    public static class TestRerankingActionResponse extends ActionResponse {

        private final List<Float> scores;

        public TestRerankingActionResponse(List<Float> scores) {
            super();
            this.scores = scores;
        }

        public TestRerankingActionResponse(StreamInput in) throws IOException {
            super(in);
            this.scores = in.readCollectionAsList(StreamInput::readFloat);
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            out.writeCollection(scores, StreamOutput::writeFloat);
        }
    }

    public static class TestRerankingTransportAction extends HandledTransportAction<
        TestRerankingActionRequest,
        TestRerankingActionResponse> {
        @Inject
        public TestRerankingTransportAction(TransportService transportService, ActionFilters actionFilters) {
            super(
                TEST_RERANKING_ACTION_TYPE.name(),
                transportService,
                actionFilters,
                TestRerankingActionRequest::new,
                EsExecutors.DIRECT_EXECUTOR_SERVICE
            );
        }

        @Override
        protected void doExecute(Task task, TestRerankingActionRequest request, ActionListener<TestRerankingActionResponse> listener) {
            List<String> featureData = request.docFeatures;
            List<Float> scores = featureData.stream().map(Float::parseFloat).toList();
            listener.onResponse(new TestRerankingActionResponse(scores));
        }
    }

    public static class TestRerankingRankFeaturePhaseRankCoordinatorContext extends RequestRankFeaturePhaseRankCoordinatorContext<
        TestRerankingActionRequest,
        TestRerankingActionResponse> {

        TestRerankingRankFeaturePhaseRankCoordinatorContext(
            int size,
            int from,
            int windowSize,
            Client client,
            String inferenceId,
            String inferenceText
        ) {
            super(size, from, windowSize, client, inferenceId, inferenceText);
        }

        @Override
        protected TestRerankingActionRequest generateRequest(List<String> docFeatures) {
            return new TestRerankingActionRequest(docFeatures);
        }

        @Override
        protected ActionType<TestRerankingActionResponse> actionType() {
            return TEST_RERANKING_ACTION_TYPE;
        }

        @Override
        protected float[] extractScoresFromResponse(TestRerankingActionResponse response) {
            float[] scores = new float[response.scores.size()];
            for (int i = 0; i < response.scores.size(); i++) {
                scores[i] = response.scores.get(i);
            }
            return scores;
        }
    }

    public static class RerankerBasedRankBuilder extends RankBuilder {

        public static final ParseField FIELD_FIELD = new ParseField("field");
        public static final ParseField INFERENCE_ID = new ParseField("inference_id");
        public static final ParseField INFERENCE_TEXT = new ParseField("inference_text");
        static final ConstructingObjectParser<RerankerBasedRankBuilder, Void> PARSER = new ConstructingObjectParser<>(
            "field-based-rank",
            args -> {
                int rankWindowSize = args[0] == null ? DEFAULT_RANK_WINDOW_SIZE : (int) args[0];
                String field = (String) args[1];
                if (field == null || field.isEmpty()) {
                    throw new IllegalArgumentException("Field cannot be null or empty");
                }
                final String inferenceId = (String) args[2];
                final String inferenceText = (String) args[3];
                return new RerankerBasedRankBuilder(rankWindowSize, field, inferenceId, inferenceText);
            }
        );

        static {
            PARSER.declareInt(optionalConstructorArg(), RANK_WINDOW_SIZE_FIELD);
            PARSER.declareString(constructorArg(), FIELD_FIELD);
            PARSER.declareString(constructorArg(), INFERENCE_ID);
            PARSER.declareString(constructorArg(), INFERENCE_TEXT);
        }

        protected final String field;
        protected final String inferenceId;
        protected final String inferenceText;

        public static RerankerBasedRankBuilder fromXContent(XContentParser parser) throws IOException {
            return PARSER.parse(parser, null);
        }

        public RerankerBasedRankBuilder(
            final int rankWindowSize,
            final String field,
            final String inferenceId,
            final String inferenceText
        ) {
            super(rankWindowSize);
            this.field = field;
            this.inferenceId = inferenceId;
            this.inferenceText = inferenceText;
        }

        public RerankerBasedRankBuilder(StreamInput in) throws IOException {
            super(in);
            this.field = in.readString();
            this.inferenceId = in.readString();
            this.inferenceText = in.readString();
        }

        @Override
        protected void doWriteTo(StreamOutput out) throws IOException {
            out.writeString(field);
            out.writeString(inferenceId);
            out.writeString(inferenceText);
        }

        @Override
        protected void doXContent(XContentBuilder builder, Params params) throws IOException {
            builder.field(FIELD_FIELD.getPreferredName(), field);
            builder.field(INFERENCE_ID.getPreferredName(), inferenceId);
            builder.field(INFERENCE_TEXT.getPreferredName(), inferenceText);
        }

        @Override
        public boolean isCompoundBuilder() {
            return false;
        }

        @Override
        public QueryPhaseRankShardContext buildQueryPhaseShardContext(List<Query> queries, int from) {
            return new RerankingQueryPhaseRankShardContext(queries, rankWindowSize());
        }

        @Override
        public QueryPhaseRankCoordinatorContext buildQueryPhaseCoordinatorContext(int size, int from) {
            return new RerankingQueryPhaseRankCoordinatorContext(rankWindowSize());
        }

        @Override
        public RankFeaturePhaseRankShardContext buildRankFeaturePhaseShardContext() {
            return new RerankingRankFeaturePhaseRankShardContext(field);
        }

        @Override
        public RankFeaturePhaseRankCoordinatorContext buildRankFeaturePhaseCoordinatorContext(int size, int from, Client client) {
            return new TestRerankingRankFeaturePhaseRankCoordinatorContext(
                size,
                from,
                rankWindowSize(),
                client,
                inferenceId,
                inferenceText
            );
        }

        @Override
        protected boolean doEquals(RankBuilder other) {
            return other instanceof RerankerBasedRankBuilder && Objects.equals(field, ((RerankerBasedRankBuilder) other).field);
        }

        @Override
        protected int doHashCode() {
            return Objects.hash(field);
        }

        @Override
        public String getWriteableName() {
            return "inference-based-rank";
        }

        @Override
        public TransportVersion getMinimalSupportedVersion() {
            return TransportVersion.current();
        }
    }
}
