/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank.rerank;

import org.elasticsearch.TransportVersion;
import org.elasticsearch.TransportVersions;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.search.rank.RankShardResult;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

public class RankShardFeatureResult implements RankShardResult {

    /**
     * The result set of RRF docs for the shard.
     */
    public final FeatureRankDoc[] featureRankDocs;

    public RankShardFeatureResult(FeatureRankDoc[] reRankedDocs) {
        this.featureRankDocs = Objects.requireNonNull(reRankedDocs);
    }

    public RankShardFeatureResult(StreamInput in) throws IOException {
        featureRankDocs = in.readArray(FeatureRankDoc::new, FeatureRankDoc[]::new);
    }

    @Override
    public String getWriteableName() {
        return "rank-shard-feature";
    }

    @Override
    public TransportVersion getMinimalSupportedVersion() {
        return TransportVersions.V_8_12_0;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeArray(featureRankDocs);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        RankShardFeatureResult that = (RankShardFeatureResult) o;
        return Arrays.equals(featureRankDocs, that.featureRankDocs);
    }

    @Override
    public int hashCode() {
        return 31 * Arrays.hashCode(featureRankDocs);
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + "{featureRankDocs=" + Arrays.toString(featureRankDocs) + '}';
    }
}
