/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.profile;

import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.search.profile.query.QueryProfileShardResult;
import org.elasticsearch.xcontent.ToXContent;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Profile results from a shard for the search phase.
 */
public class RetrieverProfileSearchResult implements Writeable {

    private final String name;

    private final long rewriteTime;

    private final SearchProfileQueryPhaseResult searchProfileShardResult;

    private final List<RetrieverProfileSearchResult> children;

    public RetrieverProfileSearchResult(
        String name,
        long rewriteTime,
        SearchProfileQueryPhaseResult searchProfileShardResult,
        List<RetrieverProfileSearchResult> children
    ) {
        this.name = name;
        this.rewriteTime = rewriteTime;
        this.searchProfileShardResult = searchProfileShardResult;
        this.children = Collections.unmodifiableList(children);
    }

    public RetrieverProfileSearchResult(StreamInput in) throws IOException {
        this.name = in.readString();
        this.rewriteTime = in.readLong();
        this.searchProfileShardResult = in.readOptionalWriteable(SearchProfileQueryPhaseResult::new);
        this.children = in.readCollectionAsList(RetrieverProfileSearchResult::new);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(name);
        out.writeLong(rewriteTime);
        out.writeOptionalWriteable(searchProfileShardResult);
        out.writeCollection(children);
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (other == null || getClass() != other.getClass()) {
            return false;
        }
        RetrieverProfileSearchResult that = (RetrieverProfileSearchResult) other;
        return Objects.equals(name, that.name)
            && Objects.equals(searchProfileShardResult, that.searchProfileShardResult)
            && Objects.equals(children, that.children);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, searchProfileShardResult, children);
    }

    public void toXContent(XContentBuilder builder, ToXContent.Params params) throws IOException {
        builder.startObject();
        builder.field("retriever_name", name);
        builder.field("tookInMillis", rewriteTime);
        if (searchProfileShardResult != null) {
            builder.field("search");
            builder.startObject();
            if (searchProfileShardResult.getSearchProfileDfsPhaseResult() != null) {
                builder.field("dfs");
                searchProfileShardResult.getSearchProfileDfsPhaseResult().toXContent(builder, params);
            }
            builder.startArray("searches");
            for (QueryProfileShardResult result : searchProfileShardResult.getQueryProfileResults()) {
                result.toXContent(builder, params);
            }
            builder.endArray();
            builder.endObject();
        }
        if (children != null && false == children.isEmpty()) {
            builder.startArray("children");
            for (RetrieverProfileSearchResult child : children) {
                child.toXContent(builder, params);
            }
            builder.endArray();
        }
        builder.endObject();
    }
}
