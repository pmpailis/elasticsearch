/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.search.rank.twophase;

import org.elasticsearch.TransportVersion;
import org.elasticsearch.TransportVersions;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.search.rank.RankShardResult;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

/**
 * The RRF results for a single shard.
 */
public class TwoPhaseRankShardResult implements RankShardResult {

    /**
     * The result set of RRF docs for the shard.
     */
    public final TwoPhaseRankDoc[] reRankedDocs;

    public TwoPhaseRankShardResult(TwoPhaseRankDoc[] reRankedDocs) {
        this.reRankedDocs = Objects.requireNonNull(reRankedDocs);
    }

    public TwoPhaseRankShardResult(StreamInput in) throws IOException {
        reRankedDocs = in.readArray(TwoPhaseRankDoc::new, TwoPhaseRankDoc[]::new);
    }

    @Override
    public String getWriteableName() {
        return "rrf";
    }

    @Override
    public TransportVersion getMinimalSupportedVersion() {
        return TransportVersions.V_8_8_0;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeArray(reRankedDocs);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TwoPhaseRankShardResult that = (TwoPhaseRankShardResult) o;
        return Arrays.equals(reRankedDocs, that.reRankedDocs);
    }

    @Override
    public int hashCode() {
        return 31 * Arrays.hashCode(reRankedDocs);
    }

    @Override
    public String toString() {
        return "RRFRankShardResult{reRankedDocs=" + Arrays.toString(reRankedDocs) + '}';
    }
}
