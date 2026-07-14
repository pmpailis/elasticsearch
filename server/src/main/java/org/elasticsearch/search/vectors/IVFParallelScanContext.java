/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import org.apache.lucene.search.TaskExecutor;

/**
 * Per-leaf context enabling within-segment parallel scanning of IVF posting lists. Carried into the codec reader on
 * the {@link IVFKnnSearchStrategy} attached to the leaf collector; a {@code null} context means the leaf must be
 * scanned serially.
 *
 * @param taskExecutor   executor used to fan posting-list scanning out to workers. Lucene's {@code TaskExecutor}
 *                       also runs tasks on the calling thread, so a saturated pool degrades to caller-runs rather
 *                       than deadlocking.
 * @param maxWorkers     upper bound on concurrent workers for this leaf (at least 2 by construction)
 * @param checkCancelled invoked between chunks of posting lists so long scans observe search cancellation/timeout
 * @param floors         min-competitive score floors that worker collectors publish to and pull from once per
 *                       posting list. This is the leaf collector-manager's {@link ScoreFloors} when one exists —
 *                       coupling workers with the leaf collector and, when cross-leaf sharing is enabled, with the
 *                       other leaves — otherwise a leaf-local instance.
 */
public record IVFParallelScanContext(TaskExecutor taskExecutor, int maxWorkers, Runnable checkCancelled, ScoreFloors floors) {}
