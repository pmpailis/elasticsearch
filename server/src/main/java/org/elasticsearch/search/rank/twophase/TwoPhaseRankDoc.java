/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.search.rank.twophase;

import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.search.rank.RankDoc;

import java.io.IOException;
import java.util.Objects;

public class TwoPhaseRankDoc extends RankDoc {

    public float secondPhaseScore;

    public TwoPhaseRankDoc(int doc, float score, int shardIndex, float secondPhaseScore) {
        super(doc, score, shardIndex);
        this.secondPhaseScore = secondPhaseScore;
    }

    public TwoPhaseRankDoc(StreamInput in) throws IOException {
        super(in);
        secondPhaseScore = in.readOptionalFloat();
    }

    @Override
    public void doWriteTo(StreamOutput out) throws IOException {
        out.writeOptionalFloat(secondPhaseScore);
    }

    @Override
    public boolean doEquals(RankDoc rd) {
        TwoPhaseRankDoc rerankedDoc = (TwoPhaseRankDoc) rd;
        return this.secondPhaseScore == rerankedDoc.secondPhaseScore;
    }

    @Override
    public int doHashCode() {
        return 31 * Objects.hash(secondPhaseScore);
    }

    @Override
    public String toString() {
        return "RerankedDoc{"
            + "rank="
            + rank
            + ", firstPhaseScore="
            + score
            + ", secondPhaseScore="
            + secondPhaseScore
            + ", doc="
            + doc
            + ", shardIndex="
            + shardIndex
            + '}';
    }

    public float getSecondPhaseScore() {
        return secondPhaseScore;
    }
}
