/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.elasticsearch.index.codec.vectors.diskbbq.PostingMetadata;

/**
 * A single posting list selected during the parallel-IVF preparation phase (Phase A) to be scored later.
 * <p>
 * It carries the codec {@link PostingMetadata} needed to position and score the posting, plus a cheap
 * estimate of how many vectors it holds. The estimate is used only to budget work across slices (Phase B);
 * the codec still reads the exact per-posting count at scoring time, so a small over-estimate is harmless.
 */
record PostingSearchTask(PostingMetadata metadata, int estimatedVectorCount) {

    /** The query-to-document-centroid similarity for this posting; used to order slices best-first. */
    float score() {
        return metadata.documentCentroidScore();
    }
}
