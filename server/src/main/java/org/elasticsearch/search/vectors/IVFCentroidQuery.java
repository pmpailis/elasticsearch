/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.util.Bits;

import java.util.concurrent.atomic.AtomicLong;

public class IVFCentroidQuery extends Query {
    public IVFCentroidQuery(String field, float[] queryVector, IVFCentroidMeta centroid, LeafReaderContext ctx, int maxVectorsPerCentroid, AtomicLong globalMinCompetitiveScore, Bits parentBitSet) {
    }

    @Override
    public String toString(String field) {
        return "";
    }

    @Override
    public void visit(QueryVisitor visitor) {

    }

    @Override
    public boolean equals(Object obj) {
        return false;
    }

    @Override
    public int hashCode() {
        return 0;
    }
}
