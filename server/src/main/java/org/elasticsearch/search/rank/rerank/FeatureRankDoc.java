/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank.rerank;

import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.search.rank.RankDoc;

import java.io.IOException;
import java.util.Objects;

/**
 * A {@link RankDoc} that contains field data to be used later by the reranker on the coordinator node.
 */
public class FeatureRankDoc extends RankDoc {

    public String featureData;

    public FeatureRankDoc(int doc, float score, int shardIndex) {
        super(doc, score, shardIndex);
    }

    public FeatureRankDoc(StreamInput in) throws IOException {
        super(in);
        featureData = in.readOptionalString();
    }

    public void featureData(String featureData) {
        this.featureData = featureData;
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        out.writeOptionalString(featureData);
    }

    @Override
    protected boolean doEquals(RankDoc rd) {
        FeatureRankDoc other = (FeatureRankDoc) rd;
        return Objects.equals(this.featureData, other.featureData);
    }

    @Override
    protected int doHashCode() {
        return Objects.hashCode(featureData);
    }
}
