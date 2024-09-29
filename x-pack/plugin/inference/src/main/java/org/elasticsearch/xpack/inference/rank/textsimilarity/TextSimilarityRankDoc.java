/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.rank.textsimilarity;

import org.apache.lucene.search.Explanation;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.search.rank.RankDoc;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.Objects;

public class TextSimilarityRankDoc extends RankDoc {

    static final String NAME = "text_similarity_rank_doc";

    // the score of this document in the original ranking
    public final float firstPhaseScore;
    // the position of this document in the original ranking
    public final int position;

    public TextSimilarityRankDoc(int doc, float score, int shardIndex, float firstPhaseScore, int position) {
        super(doc, score, shardIndex);
        this.firstPhaseScore = firstPhaseScore;
        this.position = position;
    }

    public TextSimilarityRankDoc(StreamInput in) throws IOException {
        super(in);
        firstPhaseScore = in.readFloat();
        position = in.readInt();
    }

    @Override
    public Explanation explain(Explanation[] sources, String[] queryNames) {
        return Explanation.match(1, "text similarity score: [" + score + "]", sources);
    }

    @Override
    public void doWriteTo(StreamOutput out) throws IOException {
        out.writeFloat(firstPhaseScore);
        out.writeInt(position);
    }

    @Override
    public boolean doEquals(RankDoc rd) {
        TextSimilarityRankDoc tsrd = (TextSimilarityRankDoc) rd;
        return firstPhaseScore == tsrd.firstPhaseScore && position == tsrd.position;
    }

    @Override
    public int doHashCode() {
        return Objects.hash(firstPhaseScore, position);
    }

    @Override
    public String toString() {
        return "TextSimilarityRankDoc{"
            + "rank="
            + rank
            + ", score="
            + score
            + ", firstPhaseScore="
            + firstPhaseScore
            + ", position="
            + position
            + ", doc="
            + doc
            + ", shardIndex="
            + shardIndex
            + '}';
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    protected void doToXContent(XContentBuilder builder, Params params) throws IOException {
        builder.field("firstPhaseScore", firstPhaseScore);
        builder.field("position", position);
    }
}
