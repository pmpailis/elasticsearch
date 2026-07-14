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
 * Marker for a query-level cross-segment posting scheduler, carried into the codec reader on the
 * {@link IVFKnnSearchStrategy}. The concrete type lives in the codec package (it deals in posting-level types the
 * query layer never sees); a codec reader that supports cross-segment scheduling downcasts, performs its per-leaf
 * setup, registers the leaf's ranked postings and returns without scanning — the scheduler then drives the scan of
 * all registered leaves best-first, globally.
 */
public interface IVFCrossSegmentRegistrar {}
