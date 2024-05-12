/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.rank.rrf;

import org.apache.lucene.search.Explanation;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.search.rank.RankDoc;

import java.io.IOException;
import java.util.Arrays;

/**
 * {@code RRFRankDoc} supports additional ranking information
 * required for RRF.
 */
public class RRFRankDoc extends RankDoc {

    /**
     * The position within each result set per query. The length
     * of {@code positions} is the number of queries that are part
     * of rrf ranking. If a document isn't part of a result set for a
     * specific query then the position is {@link RRFRankDoc#NO_RANK}.
     * This allows for a direct association with each query.
     */
    public final int[] positions;

    /**
     * The score for each result set per query. The length
     * of {@code positions} is the number of queries that are part
     * of rrf ranking. If a document isn't part of a result set for a
     * specific query then the score is {@code 0f}. This allows for a
     * direct association with each query.
     */
    public final float[] scores;

    public RRFRankDoc(int doc, int shardIndex, int queryCount) {
        super(doc, 0f, shardIndex);
        positions = new int[queryCount];
        Arrays.fill(positions, NO_RANK);
        scores = new float[queryCount];
    }

    public RRFRankDoc(StreamInput in) throws IOException {
        super(in);
        rank = in.readVInt();
        positions = in.readIntArray();
        scores = in.readFloatArray();
    }

    @Override
    public void doWriteTo(StreamOutput out) throws IOException {
        out.writeVInt(rank);
        out.writeIntArray(positions);
        out.writeFloatArray(scores);
    }

    @Override
    public boolean doEquals(RankDoc rd) {
        RRFRankDoc rrfrd = (RRFRankDoc) rd;
        return Arrays.equals(positions, rrfrd.positions) && Arrays.equals(scores, rrfrd.scores);
    }

    @Override
    public Explanation explain(Explanation explanation) {
        Explanation rrfExplain = Explanation.match(score, "computed for initial ranks "
            + Arrays.toString(positions)
            + " as "
            + " as [[1.0 / (10 + 3)]]");
        Explanation[] details = new Explanation[positions.length];
        for(int i = 0; i < positions.length; i++) {
            // iterate over each of the individual queries and update explanation
            if(positions[i] == NO_RANK) {
                details[i] = Explanation.noMatch("RRF score: [0]; document not part of result set");
            } else {
                details[i] = Explanation.match(rrfScore(position[i]), "RRF score: computed for query " + i + " as [[1.0 / (10 + " + positions[i] + ")]]");
            }
        }
        return explanation;
    }

    private float rrfScore(int position) {
        return 1.0f / (rankConstant + position);
    }

    @Override
    public int doHashCode() {
        int result = Arrays.hashCode(positions);
        result = 31 * result + Arrays.hashCode(scores);
        return result;
    }

    @Override
    public String toString() {
        return "RRFRankDoc{"
            + "rank="
            + rank
            + ", positions="
            + Arrays.toString(positions)
            + ", scores="
            + Arrays.toString(scores)
            + ", score="
            + score
            + ", doc="
            + doc
            + ", shardIndex="
            + shardIndex
            + '}';
    }
}
