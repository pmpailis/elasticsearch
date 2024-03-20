/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.fetch;

import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.core.RefCounted;
import org.elasticsearch.core.SimpleRefCounted;
import org.elasticsearch.search.SearchPhaseResult;
import org.elasticsearch.search.SearchShardTarget;
import org.elasticsearch.search.internal.ShardSearchContextId;
import org.elasticsearch.search.rank.RankShardResult;
import org.elasticsearch.search.rank.twophase.TwoPhaseRankShardResult;
import org.elasticsearch.transport.LeakTracker;

import java.io.IOException;

public final class RankSearchResult extends SearchPhaseResult {

    private RankShardResult rankShardResult;
    // client side counter

    private final RefCounted refCounted = LeakTracker.wrap(new SimpleRefCounted());

    public RankSearchResult() {}

    public RankSearchResult(ShardSearchContextId id, SearchShardTarget shardTarget) {
        this.contextId = id;
        setSearchShardTarget(shardTarget);
    }

    public RankSearchResult(StreamInput in) throws IOException {
        super(in);
        contextId = new ShardSearchContextId(in);
        rankShardResult = in.readOptionalWriteable(TwoPhaseRankShardResult::new);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        assert hasReferences();
        contextId.writeTo(out);
        out.writeOptionalWriteable(rankShardResult);
    }

    @Override
    public RankSearchResult rankSearchResult() {
        return this;
    }

    public void shardResult(RankShardResult shardResult) {
        assert shardResult != null;
        this.rankShardResult = shardResult;
    }

    public RankShardResult shardResult() {
        return rankShardResult;
    }

    @Override
    public void incRef() {
        refCounted.incRef();
    }

    @Override
    public boolean tryIncRef() {
        return refCounted.tryIncRef();
    }

    @Override
    public boolean decRef() {
        if (refCounted.decRef()) {
            return true;
        }
        return false;
    }

    @Override
    public boolean hasReferences() {
        return refCounted.hasReferences();
    }
}
