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
import org.elasticsearch.test.ESTestCase;

import java.util.ArrayList;
import java.util.List;

/**
 * Unit tests for {@link IVFSlicePlanner}, the pure (no-I/O) planning of how Phase-A posting lists are
 * distributed into Phase-B {@link Slice}s and which leaves are likely to be scored by more than one worker.
 * <p>
 * The planner's only inputs are the per-leaf estimated vector counts and per-posting centroid scores, so these
 * tests build {@link LeafSearchTask}s with just those fields populated (the codec reader / accept-docs fields
 * are unused by the planner and left {@code null}). The behaviours under test — big-leaf banding, small-leaf
 * merging, the leaf-per-slice cap, best-score ordering, whole-leaf mode, and the fair-share contention
 * heuristic — are exactly the decisions that must be deterministic so that the scored result never depends on
 * worker scheduling.
 */
public class IVFSlicePlannerTests extends ESTestCase {

    /** A leaf with no selected postings must contribute no slices and never crash the planner. */
    public void testEmptyProducesNoSlices() {
        assertEquals(0, IVFSlicePlanner.buildSlices(List.of(), 4, false).length);
        assertEquals(0, IVFSlicePlanner.buildSlices(List.of(leaf()), 4, false).length);
        assertEquals(0, IVFSlicePlanner.buildSlices(List.of(leaf(), leaf()), 4, false).length);
    }

    /** A single posting yields a single one-entry slice spanning exactly that posting, carrying its score. */
    public void testSinglePosting() {
        Slice[] slices = IVFSlicePlanner.buildSlices(List.of(leaf(posting(10, 1.5f))), 4, false);
        assertEquals(1, slices.length);
        assertEquals(1, slices[0].entries().length);
        SliceEntry entry = slices[0].entries()[0];
        assertEquals(0, entry.leafIndex());
        assertEquals(0, entry.postingStart());
        assertEquals(1, entry.postingEnd());
        assertEquals(1.5f, slices[0].bestScore(), 0f);
    }

    /**
     * A leaf whose vectors exceed one slice's budget is split into consecutive closest-first bands. With 12
     * equal postings and a budget of 400 vectors (parallelism 1 -&gt; 3 target slices over 1200 vectors), each
     * band should accumulate four postings, producing three single-leaf slices that tile the leaf with no gaps.
     */
    public void testBigLeafSplitsIntoConsecutiveBands() {
        PostingSearchTask[] postings = new PostingSearchTask[12];
        for (int i = 0; i < postings.length; i++) {
            postings[i] = posting(100, 12 - i); // descending score, closest-first
        }
        Slice[] slices = IVFSlicePlanner.buildSlices(List.of(leaf(postings)), 1, false);

        assertEquals(3, slices.length);
        for (Slice slice : slices) {
            assertEquals("a band must reference exactly one leaf", 1, slice.entries().length);
            assertEquals(0, slice.entries()[0].leafIndex());
        }
        assertTilesLeaf(slices, 0, postings.length);
    }

    /**
     * Several leaves each smaller than the budget are merged into shared slices. Six one-posting leaves of ten
     * vectors with a budget of 20 (parallelism 1 -&gt; 3 target slices over 60 vectors) should pair up into three
     * two-leaf slices.
     */
    public void testSmallLeavesAreMerged() {
        List<LeafSearchTask> leaves = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            leaves.add(leaf(posting(10, 6 - i)));
        }
        Slice[] slices = IVFSlicePlanner.buildSlices(leaves, 1, false);

        assertEquals(3, slices.length);
        for (Slice slice : slices) {
            assertEquals("two ten-vector leaves fill the twenty-vector budget", 2, slice.entries().length);
        }
        assertEachLeafCoveredOnce(leaves, slices);
    }

    /**
     * Even when the vector budget is large enough to hold more, no slice may merge more than
     * {@link IVFSlicePlanner#MAX_LEAVES_PER_SLICE} leaves. Thirty ten-vector leaves give a budget of 100 (so the
     * budget alone would allow ten leaves per slice); the cap must seal each slice at eight leaves instead.
     */
    public void testMaxLeavesPerSliceCapBinds() {
        List<LeafSearchTask> leaves = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            leaves.add(leaf(posting(10, randomFloat())));
        }
        Slice[] slices = IVFSlicePlanner.buildSlices(leaves, 1, false);

        boolean sawCappedSlice = false;
        for (Slice slice : slices) {
            assertTrue(
                "slice merged " + slice.entries().length + " leaves, exceeding the cap",
                slice.entries().length <= IVFSlicePlanner.MAX_LEAVES_PER_SLICE
            );
            sawCappedSlice |= slice.entries().length == IVFSlicePlanner.MAX_LEAVES_PER_SLICE;
        }
        assertTrue("expected at least one slice sealed exactly at the leaf cap", sawCappedSlice);
        assertEachLeafCoveredOnce(leaves, slices);
    }

    /** Slices must be ordered by best centroid score descending so the shared frontier tightens earliest. */
    public void testSlicesSortedByBestScoreDescending() {
        List<LeafSearchTask> leaves = new ArrayList<>();
        for (int i = 0; i < randomIntBetween(5, 40); i++) {
            int numPostings = randomIntBetween(1, 6);
            PostingSearchTask[] postings = new PostingSearchTask[numPostings];
            for (int p = 0; p < numPostings; p++) {
                postings[p] = posting(randomIntBetween(1, 500), randomFloat() * 100);
            }
            leaves.add(leaf(postings));
        }
        Slice[] slices = IVFSlicePlanner.buildSlices(leaves, randomIntBetween(1, 8), randomBoolean());
        for (int i = 1; i < slices.length; i++) {
            assertTrue("slices must be best-score-descending", slices[i - 1].bestScore() >= slices[i].bestScore());
        }
    }

    /**
     * Whole-leaf mode (required by nested diversification) emits exactly one slice per non-empty leaf spanning
     * all of that leaf's postings, with no banding or merging; empty leaves are skipped.
     */
    public void testWholeLeafModeEmitsOneSlicePerLeaf() {
        List<LeafSearchTask> leaves = List.of(
            leaf(posting(100, 1f), posting(100, 2f)),
            leaf(), // empty -> skipped
            leaf(posting(50, 9f), posting(50, 3f), posting(50, 7f)),
            leaf(posting(10, 5f))
        );
        Slice[] slices = IVFSlicePlanner.buildSlices(leaves, 4, true);

        assertEquals("one slice per non-empty leaf", 3, slices.length);
        for (Slice slice : slices) {
            assertEquals("whole-leaf mode must not band or merge", 1, slice.entries().length);
            SliceEntry entry = slice.entries()[0];
            assertEquals("a whole-leaf slice must span the entire leaf", 0, entry.postingStart());
            assertEquals(leaves.get(entry.leafIndex()).postings().length, entry.postingEnd());
        }
        for (int i = 1; i < slices.length; i++) {
            assertTrue(slices[i - 1].bestScore() >= slices[i].bestScore());
        }
        assertEachLeafCoveredOnce(leaves, slices);
    }

    /**
     * Across a random mix of big and small leaves, every selected posting must be scored by exactly one slice
     * entry — no posting dropped, none scored twice. This is the core correctness guarantee of the distribution.
     */
    public void testEveryPostingCoveredExactlyOnce() {
        List<LeafSearchTask> leaves = new ArrayList<>();
        int numLeaves = randomIntBetween(1, 25);
        for (int i = 0; i < numLeaves; i++) {
            int numPostings = randomIntBetween(0, 10);
            PostingSearchTask[] postings = new PostingSearchTask[numPostings];
            for (int p = 0; p < numPostings; p++) {
                postings[p] = posting(randomIntBetween(1, 1000), randomFloat());
            }
            leaves.add(leaf(postings));
        }
        Slice[] slices = IVFSlicePlanner.buildSlices(leaves, randomIntBetween(1, 12), false);
        assertEachLeafCoveredOnce(leaves, slices);
    }

    /** A leaf holding more than its fair share of vectors is contended; the small leaves alongside it are not. */
    public void testMarkContendedFlagsOversizedLeaf() {
        List<LeafSearchTask> leaves = List.of(leaf(posting(1000, 1f)), leaf(posting(10, 1f)), leaf(posting(10, 1f)));
        boolean[] contended = IVFSlicePlanner.markContendedLeaves(leaves, 4);
        assertTrue("the 1000-vector leaf exceeds the fair share and is contended", contended[0]);
        assertFalse(contended[1]);
        assertFalse(contended[2]);
    }

    /** Perfectly balanced leaves never exceed the fair share, so none is marked contended. */
    public void testMarkContendedBalancedLeavesNotContended() {
        List<LeafSearchTask> leaves = List.of(
            leaf(posting(100, 1f)),
            leaf(posting(100, 1f)),
            leaf(posting(100, 1f)),
            leaf(posting(100, 1f))
        );
        boolean[] contended = IVFSlicePlanner.markContendedLeaves(leaves, 4);
        for (boolean c : contended) {
            assertFalse("equal leaves split evenly across workers are not contended", c);
        }
    }

    /** With a single worker the fair share equals the total, so no individual leaf can exceed it. */
    public void testMarkContendedSingleWorkerNeverContended() {
        List<LeafSearchTask> leaves = List.of(leaf(posting(1000, 1f)), leaf(posting(1, 1f)), leaf(posting(500, 1f)));
        boolean[] contended = IVFSlicePlanner.markContendedLeaves(leaves, 1);
        for (boolean c : contended) {
            assertFalse(c);
        }
    }

    // --- helpers -------------------------------------------------------------------------------------------

    private static PostingSearchTask posting(int estimatedVectorCount, float score) {
        return new PostingSearchTask(new PostingMetadata(0L, 0L, 0, score), estimatedVectorCount);
    }

    private static LeafSearchTask leaf(PostingSearchTask... postings) {
        return LeafSearchTask.ivf(null, null, null, null, null, null, postings, 0f);
    }

    /** Asserts that the single-leaf {@code slices} tile {@code [0, postingCount)} of {@code leafIndex} with no gaps or overlaps. */
    private static void assertTilesLeaf(Slice[] slices, int leafIndex, int postingCount) {
        boolean[] covered = new boolean[postingCount];
        int expectedNextStart = 0;
        // bands are emitted in posting order for a big leaf, so they should be contiguous and ascending
        for (Slice slice : slices) {
            for (SliceEntry entry : slice.entries()) {
                if (entry.leafIndex() != leafIndex) {
                    continue;
                }
                assertEquals("bands must be consecutive", expectedNextStart, entry.postingStart());
                for (int p = entry.postingStart(); p < entry.postingEnd(); p++) {
                    assertFalse("posting " + p + " covered twice", covered[p]);
                    covered[p] = true;
                }
                expectedNextStart = entry.postingEnd();
            }
        }
        for (int p = 0; p < postingCount; p++) {
            assertTrue("posting " + p + " never scored", covered[p]);
        }
    }

    /** Asserts that across all {@code slices} every posting of every leaf is referenced by exactly one slice entry. */
    private static void assertEachLeafCoveredOnce(List<LeafSearchTask> leaves, Slice[] slices) {
        boolean[][] covered = new boolean[leaves.size()][];
        for (int i = 0; i < leaves.size(); i++) {
            covered[i] = new boolean[leaves.get(i).postings().length];
        }
        for (Slice slice : slices) {
            for (SliceEntry entry : slice.entries()) {
                for (int p = entry.postingStart(); p < entry.postingEnd(); p++) {
                    assertFalse("leaf " + entry.leafIndex() + " posting " + p + " covered twice", covered[entry.leafIndex()][p]);
                    covered[entry.leafIndex()][p] = true;
                }
            }
        }
        for (int i = 0; i < leaves.size(); i++) {
            for (int p = 0; p < covered[i].length; p++) {
                assertTrue("leaf " + i + " posting " + p + " never scored", covered[i][p]);
            }
        }
    }
}
