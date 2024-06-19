/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank.feature;

import org.apache.lucene.search.Explanation;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.search.rank.RankDoc;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * A {@link RankDoc} that contains field data to be used later by the reranker on the coordinator node.
 */
public class RankFeatureDoc extends RankDoc {

    public static final String NAME = "rank_feature_doc";

    public Map<String, Object> featureData;
    public Object[] fields;

    public RankFeatureDoc(int doc, float score, int shardIndex) {
        super(doc, score, shardIndex);
        this.featureData = new HashMap<>();
    }

    public RankFeatureDoc(StreamInput in) throws IOException {
        super(in);
        featureData = in.readGenericMap();
        fields = in.readOptionalArray(StreamInput::readGenericValue, Object[]::new);
    }

    public void featureData(String key, Object val) {
        this.featureData.put(key, val);
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        out.writeGenericMap(featureData);
        out.writeOptionalArray(StreamOutput::writeGenericValue, fields);
    }

    @Override
    protected boolean doEquals(RankDoc rd) {
        RankFeatureDoc other = (RankFeatureDoc) rd;
        return Objects.equals(this.featureData, other.featureData) && Arrays.equals(this.fields, other.fields);
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(featureData, Objects.hash(fields));
    }

    @Override
    public Explanation explain() {
        return Explanation.noMatch("No match");
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }
}
