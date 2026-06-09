/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import java.util.ArrayList;
import java.util.List;

/**
 * Pure, side-effect-free planning of how the posting lists selected in Phase A are distributed into
 * {@link Slice}s for the parallel scoring phase (Phase B), and which leaves are likely to be scored
 * concurrently by more than one worker.
 * <p>
 * The slicing balances two goals: enough slices to keep every worker busy (oversubscription relative to the
 * available parallelism), and roughly equal vector counts per slice so no single worker becomes a straggler.
 * A leaf larger than one slice's vector budget is split into consecutive closest-first bands; several small
 * leaves are merged into one slice. All decisions are deterministic functions of the prepared leaves, so the
 * scored result does not depend on worker scheduling.
 */
final class IVFSlicePlanner {

    /** How many slices to target per unit of parallelism, so faster workers can steal extra slices. */
    static final int SLICE_OVERSUBSCRIPTION = 3;

    /** Upper bound on how many small leaves may be merged into a single slice. */
    static final int MAX_LEAVES_PER_SLICE = 8;

    private IVFSlicePlanner() {}

    /**
     * Builds the slices to be drained by Phase B workers, ordered best-centroid-score first.
     *
     * @param ivfLeaves     the prepared IVF leaves (index of a leaf in this list is its {@link SliceEntry#leafIndex()})
     * @param parallelism   the available parallelism, used to size the number of slices
     * @param wholeLeafOnly when {@code true}, emit exactly one slice per non-empty leaf with no banding or merging
     *                      (required when scoring must keep each leaf within a single collector, e.g. nested diversification)
     */
    static Slice[] buildSlices(List<LeafSearchTask> ivfLeaves, int parallelism, boolean wholeLeafOnly) {
        if (wholeLeafOnly) {
            return buildWholeLeafSlices(ivfLeaves);
        }
        long totalVectors = 0;
        int totalPostings = 0;
        for (LeafSearchTask leaf : ivfLeaves) {
            totalVectors += sumVectors(leaf.postings(), 0, leaf.postings().length);
            totalPostings += leaf.postings().length;
        }
        if (totalPostings == 0) {
            return new Slice[0];
        }
        int targetSlices = (int) Math.max(1, Math.min(totalPostings, (long) Math.max(1, parallelism) * SLICE_OVERSUBSCRIPTION));
        long budget = Math.max(1, (long) Math.ceil((double) totalVectors / targetSlices));

        List<Slice> slices = new ArrayList<>(targetSlices);
        List<SliceEntry> pending = new ArrayList<>();
        long pendingVectors = 0;
        float pendingBest = Float.NEGATIVE_INFINITY;

        for (int leafIndex = 0; leafIndex < ivfLeaves.size(); leafIndex++) {
            PostingSearchTask[] postings = ivfLeaves.get(leafIndex).postings();
            if (postings.length == 0) {
                continue;
            }
            long leafVectors = sumVectors(postings, 0, postings.length);
            if (leafVectors > budget) {
                // Big leaf: seal any pending small-leaf group first, then split into consecutive ~budget bands.
                pendingVectors = flushPending(slices, pending, pendingVectors, pendingBest);
                pendingBest = Float.NEGATIVE_INFINITY;
                int bandStart = 0;
                long bandVectors = 0;
                for (int postingIndex = 0; postingIndex < postings.length; postingIndex++) {
                    bandVectors += postings[postingIndex].estimatedVectorCount();
                    boolean last = postingIndex == postings.length - 1;
                    if (bandVectors >= budget || last) {
                        int bandEnd = postingIndex + 1;
                        slices.add(
                            new Slice(
                                new SliceEntry[] { new SliceEntry(leafIndex, bandStart, bandEnd) },
                                bestScore(postings, bandStart, bandEnd)
                            )
                        );
                        bandStart = bandEnd;
                        bandVectors = 0;
                    }
                }
            } else {
                // Small leaf: greedily merge with previous small leaves, sealing when the budget or leaf cap is hit.
                if (pendingVectors + leafVectors > budget || pending.size() >= MAX_LEAVES_PER_SLICE) {
                    pendingVectors = flushPending(slices, pending, pendingVectors, pendingBest);
                    pendingBest = Float.NEGATIVE_INFINITY;
                }
                pending.add(new SliceEntry(leafIndex, 0, postings.length));
                pendingVectors += leafVectors;
                pendingBest = Math.max(pendingBest, bestScore(postings, 0, postings.length));
            }
        }
        flushPending(slices, pending, pendingVectors, pendingBest);
        return sortByBestScoreDesc(slices);
    }

    /**
     * Marks each leaf as likely-contended (scored by more than one worker) or not. A contended leaf should be
     * scored through a thread-confined posting-list mapping (no cross-thread session contention); a leaf scored
     * by at most one worker can simply clone the shared warm mapping. The decision is a fair-share heuristic and
     * only affects performance: every worker holds its own handle regardless, so correctness does not depend on it.
     */
    static boolean[] markContendedLeaves(List<LeafSearchTask> ivfLeaves, int workerCount) {
        boolean[] contended = new boolean[ivfLeaves.size()];
        long[] leafVectors = new long[ivfLeaves.size()];
        long totalVectors = 0;
        for (int leafIndex = 0; leafIndex < ivfLeaves.size(); leafIndex++) {
            PostingSearchTask[] postings = ivfLeaves.get(leafIndex).postings();
            leafVectors[leafIndex] = sumVectors(postings, 0, postings.length);
            totalVectors += leafVectors[leafIndex];
        }
        long fairShare = totalVectors / Math.max(1, workerCount);
        for (int leafIndex = 0; leafIndex < ivfLeaves.size(); leafIndex++) {
            contended[leafIndex] = leafVectors[leafIndex] > fairShare;
        }
        return contended;
    }

    private static Slice[] buildWholeLeafSlices(List<LeafSearchTask> ivfLeaves) {
        List<Slice> slices = new ArrayList<>(ivfLeaves.size());
        for (int leafIndex = 0; leafIndex < ivfLeaves.size(); leafIndex++) {
            PostingSearchTask[] postings = ivfLeaves.get(leafIndex).postings();
            if (postings.length == 0) {
                continue;
            }
            slices.add(
                new Slice(new SliceEntry[] { new SliceEntry(leafIndex, 0, postings.length) }, bestScore(postings, 0, postings.length))
            );
        }
        return sortByBestScoreDesc(slices);
    }

    private static long flushPending(List<Slice> slices, List<SliceEntry> pending, long pendingVectors, float pendingBest) {
        if (pending.isEmpty() == false) {
            slices.add(new Slice(pending.toArray(new SliceEntry[0]), pendingBest));
            pending.clear();
        }
        return 0;
    }

    private static Slice[] sortByBestScoreDesc(List<Slice> slices) {
        Slice[] array = slices.toArray(new Slice[0]);
        // Sort highest-centroid-score first so the shared competitive frontier tightens as early as possible.
        java.util.Arrays.sort(array, (a, b) -> Float.compare(b.bestScore(), a.bestScore()));
        return array;
    }

    private static long sumVectors(PostingSearchTask[] postings, int start, int end) {
        long sum = 0;
        for (int i = start; i < end; i++) {
            sum += postings[i].estimatedVectorCount();
        }
        return sum;
    }

    private static float bestScore(PostingSearchTask[] postings, int start, int end) {
        float best = Float.NEGATIVE_INFINITY;
        for (int i = start; i < end; i++) {
            best = Math.max(best, postings[i].score());
        }
        return best;
    }
}
