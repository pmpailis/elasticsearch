/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.dfs;

import org.apache.lucene.search.ScoreDoc;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.common.lucene.Lucene;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.search.vectors.VectorData;

import java.io.IOException;

/**
 * Carries per-KNN-search rescore instructions from the coordinator to a shard during the query phase.
 * Each instance represents one KNN search that needs float-vector rescoring on this shard.
 */
public class DfsKnnRescoreInfo implements Writeable {
    private final ScoreDoc[] scoreDocs;
    private final String fieldName;
    private final VectorData queryVector;
    private final int k;
    private final float boost;
    @Nullable
    private final String queryName;

    public DfsKnnRescoreInfo(ScoreDoc[] scoreDocs, String fieldName, VectorData queryVector, int k, float boost, @Nullable String queryName) {
        this.scoreDocs = scoreDocs;
        this.fieldName = fieldName;
        this.queryVector = queryVector;
        this.k = k;
        this.boost = boost;
        this.queryName = queryName;
    }

    public DfsKnnRescoreInfo(StreamInput in) throws IOException {
        this.scoreDocs = in.readArray(Lucene::readScoreDoc, ScoreDoc[]::new);
        this.fieldName = in.readString();
        this.queryVector = new VectorData(in);
        this.k = in.readVInt();
        this.boost = in.readFloat();
        this.queryName = in.readOptionalString();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeArray(Lucene::writeScoreDoc, scoreDocs);
        out.writeString(fieldName);
        queryVector.writeTo(out);
        out.writeVInt(k);
        out.writeFloat(boost);
        out.writeOptionalString(queryName);
    }

    public ScoreDoc[] scoreDocs() {
        return scoreDocs;
    }

    public String fieldName() {
        return fieldName;
    }

    public VectorData queryVector() {
        return queryVector;
    }

    public int k() {
        return k;
    }

    public float boost() {
        return boost;
    }

    @Nullable
    public String queryName() {
        return queryName;
    }
}
