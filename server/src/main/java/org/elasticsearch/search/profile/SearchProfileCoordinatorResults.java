/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.profile;

import org.elasticsearch.common.Strings;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.xcontent.ToXContent;
import org.elasticsearch.xcontent.ToXContentFragment;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.Map;
import java.util.Objects;

public class SearchProfileCoordinatorResults implements Writeable, ToXContentFragment {

    private final String nodeId;
    private final Map<String, Long> breakdownMap;

    public SearchProfileCoordinatorResults(String nodeId, Map<String, Long> breakdownMap) {
        this.nodeId = nodeId;
        this.breakdownMap = breakdownMap;
    }

    public SearchProfileCoordinatorResults(StreamInput in) throws IOException {
        nodeId = in.readString();
        breakdownMap = in.readMap(StreamInput::readString, StreamInput::readLong);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(nodeId);
        out.writeMap(breakdownMap, StreamOutput::writeString, StreamOutput::writeLong);
    }

    public String getNodeId() {
        return this.nodeId;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, ToXContent.Params params) throws IOException {
        builder.field("node_id", nodeId);
        if (false == breakdownMap.isEmpty()) {
            builder.field("breakdown", breakdownMap);
        }
        return builder;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SearchProfileCoordinatorResults that = (SearchProfileCoordinatorResults) o;
        return nodeId.equals(that.nodeId) && Objects.equals(breakdownMap, that.breakdownMap);
    }

    @Override
    public int hashCode() {
        return Objects.hash(nodeId, breakdownMap);
    }

    @Override
    public String toString() {
        return Strings.toString(this);
    }
}
