/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

/**
 * A contiguous run of posting lists within a single leaf, scored as {@code postings[postingStart, postingEnd)}
 * of the {@code leafIndex}-th prepared {@link LeafSearchTask}. {@code postingStart} is inclusive and
 * {@code postingEnd} exclusive.
 */
record SliceEntry(int leafIndex, int postingStart, int postingEnd) {}
