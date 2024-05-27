/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank.rerank;

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
import org.elasticsearch.search.SearchHits;
import org.elasticsearch.search.rank.RankBuilder;
import org.elasticsearch.search.rank.RankShardResult;
import org.elasticsearch.search.rank.context.QueryPhaseRankCoordinatorContext;
import org.elasticsearch.search.rank.context.QueryPhaseRankShardContext;
import org.elasticsearch.search.rank.context.RankFeaturePhaseRankCoordinatorContext;
import org.elasticsearch.search.rank.context.RankFeaturePhaseRankShardContext;
import org.elasticsearch.search.rank.feature.RankFeatureShardResult;
import org.elasticsearch.tasks.Task;
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

import static org.elasticsearch.xcontent.ConstructingObjectParser.constructorArg;
import static org.elasticsearch.xcontent.ConstructingObjectParser.optionalConstructorArg;

public class MockedRequestActionBasedRerankerIT extends AbstractRerankerIT {

    @Override
    public void testNonBlockingRanking() {
        // no-op
    }

    private static final TestRerankingActionType TEST_RERANKING_ACTION_TYPE = new TestRerankingActionType("internal:test_reranking_action");

    private static final String inferenceId = "inference-id";
    private static final String inferenceText = "inference-text";

    @Override
    protected RankBuilder getRankBuilder(int rankWindowSize, String rankFeatureField) {
        return new MockRequestActionBasedRankBuilder(rankWindowSize, rankFeatureField, inferenceId, inferenceText);
    }

    @Override
    protected RankBuilder getShardThrowingRankBuilder(int rankWindowSize, String rankFeatureField) {
        return new ShardThrowingMockRequestActionBasedRankBuilder(rankWindowSize, rankFeatureField, inferenceId, inferenceText);
    }

    @Override
    protected RankBuilder getCoordinatorThrowingRankBuilder(int rankWindowSize, String rankFeatureField) {
        return new CoordinatorThrowingMockRequestActionBasedRankBuilder(rankWindowSize, rankFeatureField, inferenceId, inferenceText);
    }

    @Override
    protected Collection<Class<? extends Plugin>> pluginsNeeded() {
        return List.of(RerankerServicePlugin.class, RequestActionBasedRerankerPlugin.class);
    }

    public static class RerankerServicePlugin extends Plugin implements ActionPlugin {

        @Override
        public Collection<ActionHandler<? extends ActionRequest, ? extends ActionResponse>> getActions() {
            return List.of(new ActionHandler<>(TEST_RERANKING_ACTION_TYPE, TestRerankingTransportAction.class));
        }
    }

    public static class RequestActionBasedRerankerPlugin extends Plugin implements SearchPlugin {

        private static final String REQUEST_ACTION_BASED_RANK_BUILDER_NAME = "request-action-based-rank";
        private static final String SHARD_THROWING_REQUEST_ACTION_BASED_RANK_BUILDER_NAME = "shard-throwing-request-action-based-rank";
        private static final String COORDINATOR_THROWING_REQUEST_ACTION_BASED_RANK_BUILDER_NAME =
            "coordinator-throwing-request-action-based-rank";

        @Override
        public List<NamedWriteableRegistry.Entry> getNamedWriteables() {
            return List.of(
                new NamedWriteableRegistry.Entry(
                    RankBuilder.class,
                    REQUEST_ACTION_BASED_RANK_BUILDER_NAME,
                    MockRequestActionBasedRankBuilder::new
                ),
                new NamedWriteableRegistry.Entry(
                    RankBuilder.class,
                    SHARD_THROWING_REQUEST_ACTION_BASED_RANK_BUILDER_NAME,
                    ShardThrowingMockRequestActionBasedRankBuilder::new
                ),
                new NamedWriteableRegistry.Entry(
                    RankBuilder.class,
                    COORDINATOR_THROWING_REQUEST_ACTION_BASED_RANK_BUILDER_NAME,
                    CoordinatorThrowingMockRequestActionBasedRankBuilder::new
                ),
                new NamedWriteableRegistry.Entry(RankShardResult.class, "rank-feature-shard", RankFeatureShardResult::new)
            );
        }

        @Override
        public List<NamedXContentRegistry.Entry> getNamedXContent() {
            return List.of(
                new NamedXContentRegistry.Entry(
                    RankBuilder.class,
                    new ParseField(REQUEST_ACTION_BASED_RANK_BUILDER_NAME),
                    MockRequestActionBasedRankBuilder::fromXContent
                ),
                new NamedXContentRegistry.Entry(
                    RankBuilder.class,
                    new ParseField(SHARD_THROWING_REQUEST_ACTION_BASED_RANK_BUILDER_NAME),
                    ShardThrowingMockRequestActionBasedRankBuilder::fromXContent
                ),
                new NamedXContentRegistry.Entry(
                    RankBuilder.class,
                    new ParseField(COORDINATOR_THROWING_REQUEST_ACTION_BASED_RANK_BUILDER_NAME),
                    CoordinatorThrowingMockRequestActionBasedRankBuilder::fromXContent
                )
            );
        }
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

        public boolean shouldFail() {
            return false;
        }
    }

    public static class TestThrowingRerankingActionRequest extends TestRerankingActionRequest {

        public TestThrowingRerankingActionRequest(List<String> docFeatures) {
            super(docFeatures);
        }

        public TestThrowingRerankingActionRequest(StreamInput in) throws IOException {
            super(in);
        }

        @Override
        public boolean shouldFail() {
            return true;
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
            if (request.shouldFail()) {
                throw new UnsupportedOperationException("simulated failure");
            }
            List<String> featureData = request.docFeatures;
            List<Float> scores = featureData.stream().map(Float::parseFloat).toList();
            listener.onResponse(new TestRerankingActionResponse(scores));
        }
    }

    public static class TestRerankingRankFeaturePhaseRankCoordinatorContext extends ActionRequestRankFeaturePhaseRankCoordinatorContext<
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

    public static class MockRequestActionBasedRankBuilder extends RankBuilder {

        public static final ParseField FIELD_FIELD = new ParseField("field");
        public static final ParseField INFERENCE_ID = new ParseField("inference_id");
        public static final ParseField INFERENCE_TEXT = new ParseField("inference_text");
        static final ConstructingObjectParser<MockRequestActionBasedRankBuilder, Void> PARSER = new ConstructingObjectParser<>(
            "request-action-based-rank",
            args -> {
                int rankWindowSize = args[0] == null ? DEFAULT_RANK_WINDOW_SIZE : (int) args[0];
                String field = (String) args[1];
                if (field == null || field.isEmpty()) {
                    throw new IllegalArgumentException("Field cannot be null or empty");
                }
                final String inferenceId = (String) args[2];
                final String inferenceText = (String) args[3];
                return new MockRequestActionBasedRankBuilder(rankWindowSize, field, inferenceId, inferenceText);
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

        public static MockRequestActionBasedRankBuilder fromXContent(XContentParser parser) throws IOException {
            return PARSER.parse(parser, null);
        }

        public MockRequestActionBasedRankBuilder(
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

        public MockRequestActionBasedRankBuilder(StreamInput in) throws IOException {
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
            return other instanceof MockRequestActionBasedRankBuilder
                && Objects.equals(field, ((MockRequestActionBasedRankBuilder) other).field);
        }

        @Override
        protected int doHashCode() {
            return Objects.hash(field);
        }

        @Override
        public String getWriteableName() {
            return "request-action-based-rank";
        }

        @Override
        public TransportVersion getMinimalSupportedVersion() {
            return TransportVersion.current();
        }
    }

    public static class ShardThrowingMockRequestActionBasedRankBuilder extends MockRequestActionBasedRankBuilder {

        public static final ParseField FIELD_FIELD = new ParseField("field");
        public static final ParseField INFERENCE_ID = new ParseField("inference_id");
        public static final ParseField INFERENCE_TEXT = new ParseField("inference_text");
        static final ConstructingObjectParser<ShardThrowingMockRequestActionBasedRankBuilder, Void> PARSER = new ConstructingObjectParser<>(
            "shard-throwing-request-action-based-rank",
            args -> {
                int rankWindowSize = args[0] == null ? DEFAULT_RANK_WINDOW_SIZE : (int) args[0];
                String field = (String) args[1];
                if (field == null || field.isEmpty()) {
                    throw new IllegalArgumentException("Field cannot be null or empty");
                }
                final String inferenceId = (String) args[2];
                final String inferenceText = (String) args[3];
                return new ShardThrowingMockRequestActionBasedRankBuilder(rankWindowSize, field, inferenceId, inferenceText);
            }
        );

        static {
            PARSER.declareInt(optionalConstructorArg(), RANK_WINDOW_SIZE_FIELD);
            PARSER.declareString(constructorArg(), FIELD_FIELD);
            PARSER.declareString(constructorArg(), INFERENCE_ID);
            PARSER.declareString(constructorArg(), INFERENCE_TEXT);
        }

        public static ShardThrowingMockRequestActionBasedRankBuilder fromXContent(XContentParser parser) throws IOException {
            return PARSER.parse(parser, null);
        }

        public ShardThrowingMockRequestActionBasedRankBuilder(
            final int rankWindowSize,
            final String field,
            final String inferenceId,
            final String inferenceText
        ) {
            super(rankWindowSize, field, inferenceId, inferenceText);
        }

        public ShardThrowingMockRequestActionBasedRankBuilder(StreamInput in) throws IOException {
            super(in);
        }

        @Override
        public RankFeaturePhaseRankShardContext buildRankFeaturePhaseShardContext() {
            return new RankFeaturePhaseRankShardContext(field) {
                @Override
                public RankShardResult buildRankFeatureShardResult(SearchHits hits, int shardId) {
                    throw new IllegalArgumentException("This rank builder throws an exception");
                }
            };
        }

        @Override
        public String getWriteableName() {
            return "shard-throwing-request-action-based-rank";
        }
    }

    public static class CoordinatorThrowingMockRequestActionBasedRankBuilder extends MockRequestActionBasedRankBuilder {

        public static final ParseField FIELD_FIELD = new ParseField("field");
        public static final ParseField INFERENCE_ID = new ParseField("inference_id");
        public static final ParseField INFERENCE_TEXT = new ParseField("inference_text");
        static final ConstructingObjectParser<CoordinatorThrowingMockRequestActionBasedRankBuilder, Void> PARSER =
            new ConstructingObjectParser<>("coordinator-throwing-request-action-based-rank", args -> {
                int rankWindowSize = args[0] == null ? DEFAULT_RANK_WINDOW_SIZE : (int) args[0];
                String field = (String) args[1];
                if (field == null || field.isEmpty()) {
                    throw new IllegalArgumentException("Field cannot be null or empty");
                }
                final String inferenceId = (String) args[2];
                final String inferenceText = (String) args[3];
                return new CoordinatorThrowingMockRequestActionBasedRankBuilder(rankWindowSize, field, inferenceId, inferenceText);
            });

        static {
            PARSER.declareInt(optionalConstructorArg(), RANK_WINDOW_SIZE_FIELD);
            PARSER.declareString(constructorArg(), FIELD_FIELD);
            PARSER.declareString(constructorArg(), INFERENCE_ID);
            PARSER.declareString(constructorArg(), INFERENCE_TEXT);
        }

        public static CoordinatorThrowingMockRequestActionBasedRankBuilder fromXContent(XContentParser parser) throws IOException {
            return PARSER.parse(parser, null);
        }

        public CoordinatorThrowingMockRequestActionBasedRankBuilder(
            final int rankWindowSize,
            final String field,
            final String inferenceId,
            final String inferenceText
        ) {
            super(rankWindowSize, field, inferenceId, inferenceText);
        }

        public CoordinatorThrowingMockRequestActionBasedRankBuilder(StreamInput in) throws IOException {
            super(in);
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
            ) {
                @Override
                protected TestRerankingActionRequest generateRequest(List<String> docFeatures) {
                    return new TestThrowingRerankingActionRequest(docFeatures);
                }
            };
        }

        @Override
        public String getWriteableName() {
            return "coordinator-throwing-request-action-based-rank";
        }
    }
}
