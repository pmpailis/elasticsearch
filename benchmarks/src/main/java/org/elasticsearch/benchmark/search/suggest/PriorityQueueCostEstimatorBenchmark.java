/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.benchmark.search.suggest;

import org.apache.lucene.codecs.TermStats;
import org.apache.lucene.search.spell.SuggestWord;
import org.apache.lucene.search.spell.SuggestWordQueue;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.PriorityQueue;
import org.apache.lucene.util.RamUsageEstimator;
import org.elasticsearch.search.suggest.Suggester;
import org.elasticsearch.search.suggest.phrase.Correction;
import org.elasticsearch.search.suggest.phrase.DirectCandidateGenerator;
import org.openjdk.jmh.annotations.AuxCounters;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;

/**
 * Validates {@link Suggester#priorityQueueRamBytesUsed(int)} and the phrase suggester's composed reservation.
 * <p>
 * Three scenarios are covered:
 * <ul>
 *   <li><strong>SuggestWordQueue</strong> (term suggester and phrase {@code DirectCandidateGenerator}) -
 *       {@link SuggestWordQueueState}.</li>
 *   <li><strong>Correction queue</strong> (phrase {@code CandidateScorer}'s {@code PriorityQueue}) -
 *       {@link CorrectionQueueState}. The breaker still charges {@link Suggester#priorityQueueRamBytesUsed(int)}
 *       with {@code SuggestWord}-shaped entries even though each slot holds a heavier {@link Correction} with a
 *       {@link DirectCandidateGenerator.Candidate Candidate[]} per phrase token.</li>
 *   <li><strong>Phrase reservation sum</strong> ({@code shard_size} + &Sigma; generator sizes charged by
 *       {@code PhraseSuggester}) - {@link PhraseReservationState}. Generators run sequentially at runtime, so
 *       {@code reservationOverSequentialPeakRatio} confirms the summed reservation over-budgets the sequential peak.</li>
 * </ul>
 * <p>
 * Full footprints are computed deterministically from {@link RamUsageEstimator}'s JVM-calibrated component sizes
 * rather than by walking the object graph with {@code sizeOfObject}. Cross-check allocations by running
 * {@code buildWithElements} or {@code buildCorrectionQueueWithElements} with {@code -prof gc} and reading
 * {@code gc.alloc.rate.norm}.
 * <p>
 * Example: {@code ../gradlew run --args "org.elasticsearch.benchmark.search.suggest.PriorityQueueCostEstimatorBenchmark -prof gc"}
 */
@Fork(1)
// The reported quantities are deterministic (exact byte counts / a constant-time formula), so extra iterations add no signal.
// A single measurement iteration also keeps the AuxCounters readable: AuxCounters(EVENTS) sums the published constants across
// measurement iterations, which would otherwise scale the values by the iteration count in the summary table.
@Warmup(iterations = 1)
@Measurement(iterations = 1)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@BenchmarkMode(Mode.AverageTime)
@SuppressWarnings("unused") // invoked by JMH
public class PriorityQueueCostEstimatorBenchmark {

    @State(Scope.Benchmark)
    public static class SuggestWordQueueState {

        @Param({ "1", "16", "256", "4096", "65536" })
        public int size;

        /** Length of each suggested word (ASCII, so the JVM stores it as a compact Latin1 {@code String}). */
        @Param({ "8", "24" })
        public int wordLength;

        private char[] word;
        private long precomputedEstimate;
        private long precomputedBackingArrayBytes;
        private long precomputedFullFootprintBytes;
        private double precomputedBackingRatio;
        private double precomputedFullRatio;

        @Setup(Level.Trial)
        public void setupTrial() {
            word = new char[wordLength];
            Arrays.fill(word, 'a');

            precomputedEstimate = Suggester.priorityQueueRamBytesUsed(size);
            precomputedBackingArrayBytes = RamUsageEstimator.shallowSizeOf(new Object[size + 1]);

            // Deterministic per-element cost: the SuggestWord instance plus its String (object shell + compact Latin1 byte[]).
            long stringBytes = RamUsageEstimator.shallowSizeOfInstance(String.class) + RamUsageEstimator.sizeOf(new byte[wordLength]);
            long perElementBytes = RamUsageEstimator.shallowSizeOfInstance(SuggestWord.class) + stringBytes;
            precomputedFullFootprintBytes = RamUsageEstimator.shallowSizeOfInstance(SuggestWordQueue.class) + precomputedBackingArrayBytes
                + (long) size * perElementBytes;

            precomputedBackingRatio = precomputedBackingArrayBytes == 0
                ? 0.0
                : (double) precomputedEstimate / (double) precomputedBackingArrayBytes;
            precomputedFullRatio = precomputedFullFootprintBytes == 0
                ? 0.0
                : (double) precomputedEstimate / (double) precomputedFullFootprintBytes;
        }
    }

    /** {@code size} only - {@code wordLength} does not affect the up-front backing-array allocation. */
    @State(Scope.Benchmark)
    public static class BackingArrayState {

        @Param({ "1", "16", "256", "4096", "65536" })
        public int size;
    }

    @State(Scope.Benchmark)
    public static class CorrectionQueueState {

        /** Matches phrase suggester {@code shard_size} / {@code CandidateScorer} queue capacity. */
        @Param({ "1", "16", "256", "4096" })
        public int shardSize;

        /** Number of tokens in the input phrase; each {@link Correction} holds one {@link DirectCandidateGenerator.Candidate} per token. */
        @Param({ "1", "3", "5" })
        public int phraseTokenCount;

        @Param({ "8", "24" })
        public int wordLength;

        private byte[] termBytes;
        private long precomputedEstimate;
        private long precomputedBackingArrayBytes;
        private long precomputedFullFootprintBytes;
        private double precomputedBackingRatio;
        private double precomputedFullRatio;

        @Setup(Level.Trial)
        public void setupTrial() {
            termBytes = new byte[wordLength];
            Arrays.fill(termBytes, (byte) 'a');

            precomputedEstimate = Suggester.priorityQueueRamBytesUsed(shardSize);
            precomputedBackingArrayBytes = RamUsageEstimator.shallowSizeOf(new Object[shardSize + 1]);

            long perCandidateBytes = RamUsageEstimator.shallowSizeOfInstance(DirectCandidateGenerator.Candidate.class) + RamUsageEstimator
                .shallowSizeOfInstance(TermStats.class) + RamUsageEstimator.shallowSizeOf(new BytesRef(termBytes));
            long perCorrectionBytes = RamUsageEstimator.shallowSizeOfInstance(Correction.class) + RamUsageEstimator.shallowSizeOf(
                new DirectCandidateGenerator.Candidate[phraseTokenCount]
            ) + (long) phraseTokenCount * perCandidateBytes;
            precomputedFullFootprintBytes = precomputedBackingArrayBytes + (long) shardSize * perCorrectionBytes;

            precomputedBackingRatio = precomputedBackingArrayBytes == 0
                ? 0.0
                : (double) precomputedEstimate / (double) precomputedBackingArrayBytes;
            precomputedFullRatio = precomputedFullFootprintBytes == 0
                ? 0.0
                : (double) precomputedEstimate / (double) precomputedFullFootprintBytes;
        }
    }

    @State(Scope.Benchmark)
    public static class PhraseReservationState {

        @Param({ "16", "256", "4096" })
        public int shardSize;

        @Param({ "1", "2", "3" })
        public int numGenerators;

        @Param({ "5", "10", "20" })
        public int generatorSize;

        private long precomputedPhraseReservationBytes;
        private long precomputedSequentialPeakBytes;
        private double precomputedReservationOverSequentialPeakRatio;

        @Setup(Level.Trial)
        public void setupTrial() {
            precomputedPhraseReservationBytes = Suggester.priorityQueueRamBytesUsed(shardSize) + (long) numGenerators * Suggester
                .priorityQueueRamBytesUsed(generatorSize);
            precomputedSequentialPeakBytes = Suggester.priorityQueueRamBytesUsed(shardSize) + Suggester.priorityQueueRamBytesUsed(
                generatorSize
            );
            precomputedReservationOverSequentialPeakRatio = precomputedSequentialPeakBytes == 0
                ? 0.0
                : (double) precomputedPhraseReservationBytes / (double) precomputedSequentialPeakBytes;
        }
    }

    @AuxCounters(AuxCounters.Type.EVENTS)
    @State(Scope.Thread)
    public static class SuggestWordQueueMetrics {
        public double estimatedBytes;
        public double backingArrayBytes;
        public double fullFootprintBytes;
        public double estimateOverBackingRatio;
        public double estimateOverFullRatio;
    }

    @AuxCounters(AuxCounters.Type.EVENTS)
    @State(Scope.Thread)
    public static class CorrectionQueueMetrics {
        public double estimatedBytes;
        public double backingArrayBytes;
        public double fullFootprintBytes;
        public double estimateOverBackingRatio;
        public double estimateOverFullRatio;
    }

    @AuxCounters(AuxCounters.Type.EVENTS)
    @State(Scope.Thread)
    public static class PhraseReservationMetrics {
        public double phraseReservationBytes;
        public double sequentialPeakBytes;
        public double reservationOverSequentialPeakRatio;
    }

    /** Reports the deterministic estimate-vs-real comparison for a {@link SuggestWordQueue}; timing is irrelevant. */
    @Benchmark
    public long estimateSuggestWordQueue(SuggestWordQueueState state, SuggestWordQueueMetrics metrics) {
        publishSuggestWordQueue(metrics, state);
        return Suggester.priorityQueueRamBytesUsed(state.size);
    }

    /**
     * Ground truth for the guarded up-front {@link SuggestWordQueue} allocation. Uses {@link BackingArrayState} so
     * {@code wordLength} does not create duplicate parameter combinations. Run with {@code -prof gc}.
     */
    @Benchmark
    public SuggestWordQueue buildBackingArrayOnly(BackingArrayState state) {
        return new SuggestWordQueue(state.size);
    }

    /**
     * Ground truth for a fully populated {@link SuggestWordQueue}. Run with {@code -prof gc}; {@code gc.alloc.rate.norm}
     * should match {@code fullFootprintBytes}.
     */
    @Benchmark
    public SuggestWordQueue buildWithElements(SuggestWordQueueState state) {
        SuggestWordQueue queue = new SuggestWordQueue(state.size);
        for (int i = 0; i < state.size; i++) {
            SuggestWord suggestWord = new SuggestWord();
            suggestWord.string = new String(state.word);
            suggestWord.freq = i;
            suggestWord.score = i;
            queue.insertWithOverflow(suggestWord);
        }
        return queue;
    }

    /** Reports how {@link Suggester#priorityQueueRamBytesUsed(int)} compares to a populated {@link Correction} queue. */
    @Benchmark
    public long estimateCorrectionQueue(CorrectionQueueState state, CorrectionQueueMetrics metrics) {
        publishCorrectionQueue(metrics, state);
        return Suggester.priorityQueueRamBytesUsed(state.shardSize);
    }

    /**
     * Ground truth for a fully populated {@code CandidateScorer} queue. Run with {@code -prof gc}; compare to
     * {@code fullFootprintBytes}.
     */
    @Benchmark
    public PriorityQueue<Correction> buildCorrectionQueueWithElements(CorrectionQueueState state) {
        PriorityQueue<Correction> corrections = new PriorityQueue<>(state.shardSize) {
            @Override
            protected boolean lessThan(Correction a, Correction b) {
                return a.compareTo(b) < 0;
            }
        };
        for (int i = 0; i < state.shardSize; i++) {
            DirectCandidateGenerator.Candidate[] candidates = new DirectCandidateGenerator.Candidate[state.phraseTokenCount];
            for (int t = 0; t < state.phraseTokenCount; t++) {
                BytesRef term = new BytesRef(state.termBytes);
                candidates[t] = new DirectCandidateGenerator.Candidate(term, new TermStats(1, 1), 0.5, 0.5, false);
            }
            corrections.add(new Correction(1.0 + i, candidates));
        }
        return corrections;
    }

    /** Reports the phrase suggester's summed reservation vs the sequential peak (one generator queue at a time). */
    @Benchmark
    public long estimatePhraseReservation(PhraseReservationState state, PhraseReservationMetrics metrics) {
        publishPhraseReservation(metrics, state);
        return state.precomputedPhraseReservationBytes;
    }

    private static void publishSuggestWordQueue(SuggestWordQueueMetrics metrics, SuggestWordQueueState state) {
        metrics.estimatedBytes = state.precomputedEstimate;
        metrics.backingArrayBytes = state.precomputedBackingArrayBytes;
        metrics.fullFootprintBytes = state.precomputedFullFootprintBytes;
        metrics.estimateOverBackingRatio = state.precomputedBackingRatio;
        metrics.estimateOverFullRatio = state.precomputedFullRatio;
    }

    private static void publishCorrectionQueue(CorrectionQueueMetrics metrics, CorrectionQueueState state) {
        metrics.estimatedBytes = state.precomputedEstimate;
        metrics.backingArrayBytes = state.precomputedBackingArrayBytes;
        metrics.fullFootprintBytes = state.precomputedFullFootprintBytes;
        metrics.estimateOverBackingRatio = state.precomputedBackingRatio;
        metrics.estimateOverFullRatio = state.precomputedFullRatio;
    }

    private static void publishPhraseReservation(PhraseReservationMetrics metrics, PhraseReservationState state) {
        metrics.phraseReservationBytes = state.precomputedPhraseReservationBytes;
        metrics.sequentialPeakBytes = state.precomputedSequentialPeakBytes;
        metrics.reservationOverSequentialPeakRatio = state.precomputedReservationOverSequentialPeakRatio;
    }
}
