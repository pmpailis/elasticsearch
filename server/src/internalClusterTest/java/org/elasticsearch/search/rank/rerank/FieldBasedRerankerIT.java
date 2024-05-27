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
import org.elasticsearch.TransportVersions;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.common.io.stream.NamedWriteableRegistry;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.plugins.SearchPlugin;
import org.elasticsearch.search.SearchHits;
import org.elasticsearch.search.rank.RankBuilder;
import org.elasticsearch.search.rank.RankShardResult;
import org.elasticsearch.search.rank.context.QueryPhaseRankCoordinatorContext;
import org.elasticsearch.search.rank.context.QueryPhaseRankShardContext;
import org.elasticsearch.search.rank.context.RankFeaturePhaseRankCoordinatorContext;
import org.elasticsearch.search.rank.context.RankFeaturePhaseRankShardContext;
import org.elasticsearch.search.rank.feature.RankFeatureDoc;
import org.elasticsearch.search.rank.feature.RankFeatureShardResult;
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

public class FieldBasedRerankerIT extends AbstractRerankerIT {

    @Override
    public void testNonBlockingRanking() {
        // no-op
    }

    @Override
    protected Collection<Class<? extends Plugin>> pluginsNeeded() {
        return List.of(FieldBasedRerankerPlugin.class);
    }

    @Override
    protected RankBuilder getRankBuilder(int rankWindowSize, String rankFeatureField) {
        return new FieldBasedRankBuilder(rankWindowSize, rankFeatureField);
    }

    @Override
    protected RankBuilder getShardThrowingRankBuilder(int rankWindowSize, String rankFeatureField) {
        return new ShardThrowingRankBuilder(rankWindowSize, rankFeatureField);
    }

    @Override
    protected RankBuilder getCoordinatorThrowingRankBuilder(int rankWindowSize, String rankFeatureField) {
        return new CoordinatorThrowingRankBuilder(rankWindowSize, rankFeatureField);
    }

    public static class FieldBasedRankBuilder extends RankBuilder {

        public static final ParseField FIELD_FIELD = new ParseField("field");
        static final ConstructingObjectParser<FieldBasedRankBuilder, Void> PARSER = new ConstructingObjectParser<>(
            "field-based-rank",
            args -> {
                int rankWindowSize = args[0] == null ? DEFAULT_RANK_WINDOW_SIZE : (int) args[0];
                String field = (String) args[1];
                if (field == null || field.isEmpty()) {
                    throw new IllegalArgumentException("Field cannot be null or empty");
                }
                return new FieldBasedRankBuilder(rankWindowSize, field);
            }
        );

        static {
            PARSER.declareInt(optionalConstructorArg(), RANK_WINDOW_SIZE_FIELD);
            PARSER.declareString(constructorArg(), FIELD_FIELD);
        }

        protected final String field;

        public static FieldBasedRankBuilder fromXContent(XContentParser parser) throws IOException {
            return PARSER.parse(parser, null);
        }

        public FieldBasedRankBuilder(final int rankWindowSize, final String field) {
            super(rankWindowSize);
            this.field = field;
        }

        public FieldBasedRankBuilder(StreamInput in) throws IOException {
            super(in);
            this.field = in.readString();
        }

        @Override
        protected void doWriteTo(StreamOutput out) throws IOException {
            out.writeString(field);
        }

        @Override
        protected void doXContent(XContentBuilder builder, Params params) throws IOException {
            builder.field(FIELD_FIELD.getPreferredName(), field);
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
            return new RerankingRankFeaturePhaseRankCoordinatorContext(size, from, rankWindowSize()) {
                @Override
                protected void computeScores(RankFeatureDoc[] featureDocs, ActionListener<float[]> onFinish) {
                    float[] rankScores = new float[featureDocs.length];
                    for (int i = 0; i < rankScores.length; i++) {
                        rankScores[i] = featureDocs[i].featureData == null ? 0f : Float.parseFloat(featureDocs[i].featureData);
                    }
                    onFinish.onResponse(rankScores);
                }
            };
        }

        @Override
        protected boolean doEquals(RankBuilder other) {
            return other instanceof FieldBasedRankBuilder && Objects.equals(field, ((FieldBasedRankBuilder) other).field);
        }

        @Override
        protected int doHashCode() {
            return Objects.hash(field);
        }

        @Override
        public String getWriteableName() {
            return "field-based-rank";
        }

        @Override
        public TransportVersion getMinimalSupportedVersion() {
            return TransportVersions.RANK_FEATURE_PHASE_ADDED;
        }
    }

    public static class ShardThrowingRankBuilder extends FieldBasedRankBuilder {

        public static final ParseField FIELD_FIELD = new ParseField("field");
        static final ConstructingObjectParser<ShardThrowingRankBuilder, Void> PARSER = new ConstructingObjectParser<>(
            "shard-throwing-rank",
            args -> {
                int rankWindowSize = args[0] == null ? DEFAULT_RANK_WINDOW_SIZE : (int) args[0];
                String field = (String) args[1];
                if (field == null || field.isEmpty()) {
                    throw new IllegalArgumentException("Field cannot be null or empty");
                }
                return new ShardThrowingRankBuilder(rankWindowSize, field);
            }
        );

        static {
            PARSER.declareInt(optionalConstructorArg(), RANK_WINDOW_SIZE_FIELD);
            PARSER.declareString(constructorArg(), FIELD_FIELD);
        }

        public static ShardThrowingRankBuilder fromXContent(XContentParser parser) throws IOException {
            return PARSER.parse(parser, null);
        }

        public ShardThrowingRankBuilder(final int rankWindowSize, final String field) {
            super(rankWindowSize, field);
        }

        public ShardThrowingRankBuilder(StreamInput in) throws IOException {
            super(in);
        }

        @Override
        public String getWriteableName() {
            return "shard-throwing-rank";
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
    }

    public static class CoordinatorThrowingRankBuilder extends FieldBasedRankBuilder {

        public static final ParseField FIELD_FIELD = new ParseField("field");
        static final ConstructingObjectParser<CoordinatorThrowingRankBuilder, Void> PARSER = new ConstructingObjectParser<>(
            "coordinator-throwing-rank",
            args -> {
                int rankWindowSize = args[0] == null ? DEFAULT_RANK_WINDOW_SIZE : (int) args[0];
                String field = (String) args[1];
                if (field == null || field.isEmpty()) {
                    throw new IllegalArgumentException("Field cannot be null or empty");
                }
                return new CoordinatorThrowingRankBuilder(rankWindowSize, field);
            }
        );

        static {
            PARSER.declareInt(optionalConstructorArg(), RANK_WINDOW_SIZE_FIELD);
            PARSER.declareString(constructorArg(), FIELD_FIELD);
        }

        public static CoordinatorThrowingRankBuilder fromXContent(XContentParser parser) throws IOException {
            return PARSER.parse(parser, null);
        }

        public CoordinatorThrowingRankBuilder(final int rankWindowSize, final String field) {
            super(rankWindowSize, field);
        }

        public CoordinatorThrowingRankBuilder(StreamInput in) throws IOException {
            super(in);
        }

        @Override
        public String getWriteableName() {
            return "coordinator-throwing-rank";
        }

        @Override
        public RankFeaturePhaseRankCoordinatorContext buildRankFeaturePhaseCoordinatorContext(int size, int from, Client client) {
            return new RerankingRankFeaturePhaseRankCoordinatorContext(size, from, rankWindowSize()) {
                @Override
                protected void computeScores(RankFeatureDoc[] featureDocs, ActionListener<float[]> onFinish) {
                    throw new UnsupportedOperationException("simulated-failure");
                }
            };
        }
    }

    public static class FieldBasedRerankerPlugin extends Plugin implements SearchPlugin {

        private static final String FIELD_BASED_RANK_BUILDER_NAME = "field-based-rank";
        private static final String SHARD_THROWING_RANK_BUILDER_NAME = "shard-throwing-rank";
        private static final String COORDINATOR_THROWING_RANK_BUILDER_NAME = "coordinator-throwing-rank";

        @Override
        public List<NamedWriteableRegistry.Entry> getNamedWriteables() {
            return List.of(
                new NamedWriteableRegistry.Entry(RankBuilder.class, FIELD_BASED_RANK_BUILDER_NAME, FieldBasedRankBuilder::new),
                new NamedWriteableRegistry.Entry(RankBuilder.class, SHARD_THROWING_RANK_BUILDER_NAME, ShardThrowingRankBuilder::new),
                new NamedWriteableRegistry.Entry(
                    RankBuilder.class,
                    COORDINATOR_THROWING_RANK_BUILDER_NAME,
                    CoordinatorThrowingRankBuilder::new
                ),
                new NamedWriteableRegistry.Entry(RankShardResult.class, "rank-feature-shard", RankFeatureShardResult::new)
            );
        }

        @Override
        public List<NamedXContentRegistry.Entry> getNamedXContent() {
            return List.of(
                new NamedXContentRegistry.Entry(
                    RankBuilder.class,
                    new ParseField(FIELD_BASED_RANK_BUILDER_NAME),
                    FieldBasedRankBuilder::fromXContent
                ),
                new NamedXContentRegistry.Entry(
                    RankBuilder.class,
                    new ParseField(SHARD_THROWING_RANK_BUILDER_NAME),
                    ShardThrowingRankBuilder::fromXContent
                ),
                new NamedXContentRegistry.Entry(
                    RankBuilder.class,
                    new ParseField(COORDINATOR_THROWING_RANK_BUILDER_NAME),
                    CoordinatorThrowingRankBuilder::fromXContent
                )
            );
        }
    }
}
