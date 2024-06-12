/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.core.ml.rerank;

import org.apache.lucene.search.Explanation;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.search.rank.RankDoc;

import java.io.IOException;
import java.util.Objects;

public class TextSimilarityRankDoc extends RankDoc {

    public static final String NAME = "text_similarity_rank_doc";

    private final float originalScore;

    public TextSimilarityRankDoc(int doc, float score, int shardIndex) {
        super(doc, score, shardIndex);
        this.originalScore = score;
    }

    public TextSimilarityRankDoc(StreamInput in) throws IOException {
        super(in);
        this.originalScore = in.readFloat();
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        out.writeFloat(originalScore);
    }

    @Override
    protected boolean doEquals(RankDoc rd) {
        return rd instanceof TextSimilarityRankDoc && originalScore == ((TextSimilarityRankDoc) rd).originalScore;
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(originalScore);
    }

    @Override
    public Explanation explain() {
        return Explanation.noMatch("xoxo");
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }
}
