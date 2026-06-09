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
 * A unit of scoring work drained by a single worker in Phase B: one or more {@link SliceEntry} runs (a band of
 * one large leaf, or several small leaves merged together). {@code bestScore} is the best query-to-centroid
 * score across the slice's postings; slices are scored best-first so the shared competitive frontier tightens
 * early.
 */
record Slice(SliceEntry[] entries, float bestScore) {}
